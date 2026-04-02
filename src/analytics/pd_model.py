"""
PD (Probability of Default) Prediction Model

Trains a binary classifier to predict the probability of loan default.
Uses mature loans only (Fully Paid or Charged Off / Default) to ensure
ground truth labels are available.

Model: HistGradientBoostingClassifier (histogram-based, fast on 1M+ rows)

Target:
    is_default = 1 if loan_status is Charged Off / Default
    is_default = 0 if loan_status is Fully Paid

Features:
    - Loan characteristics: amount, term, grade, interest rate
    - Borrower profile: income, DTI, employment length, home ownership
    - Credit history: FICO score, credit history months, delinquencies
    - Utilization: revolving balance, revolving utilization, inquiries
    - Census enrichment: median income, income growth rates by ZIP
"""

import pandas as pd
import numpy as np
import json
import sys
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, accuracy_score,
    confusion_matrix, precision_score, recall_score, f1_score
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database.connection import get_engine
import config


# ==================================================================
# Configuration
# ==================================================================

FULLY_PAID = 'Fully Paid'

DEFAULTED_STATUSES = [
    'Charged Off',
    'Default',
    'Does not meet the credit policy. Status:Charged Off'
]

ALL_MATURE = [FULLY_PAID] + DEFAULTED_STATUSES

# Columns used as model features
FEATURE_COLS = [
    'loan_amnt', 'funded_amnt', 'term', 'int_rate',
    'grade', 'sub_grade', 'emp_length', 'home_ownership',
    'annual_inc', 'verification_status', 'purpose', 'dti',
    'delinq_2yrs', 'fico_range_low', 'fico_range_high',
    'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal',
    'revol_util', 'total_acc', 'credit_history_months',
    # Census-enriched features
    'median_income_2024', 'housing_cost_2024',
    'income_growth_22_23', 'income_growth_23_24'
]

# Categorical columns that need Label Encoding
CATEGORICAL_COLS = ['grade', 'sub_grade', 'home_ownership',
                    'verification_status', 'purpose', 'term']

# Model output paths
MODEL_DIR = config.BASE_DIR / 'output' / 'models'
MODEL_PATH = MODEL_DIR / 'pd_model.joblib'
SCALER_PATH = MODEL_DIR / 'pd_scaler.joblib'
ENCODERS_PATH = MODEL_DIR / 'pd_encoders.joblib'

# Dashboard export paths
METRICS_PATH = MODEL_DIR / 'pd_model_metrics.json'
FEATURE_IMPORTANCE_PATH = MODEL_DIR / 'pd_feature_importance.csv'


class PDModel:
    """
    Probability of Default prediction model.

    Uses HistGradientBoostingClassifier (histogram-based gradient boosting)
    to predict the likelihood of a loan defaulting.  Outputs calibrated
    PD scores in [0, 1].

    HistGradientBoostingClassifier is chosen over GradientBoostingClassifier
    because it uses a binning strategy that reduces training time from ~1 hr
    to ~2-3 min on 1.3 M rows, with comparable AUC-ROC.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.feature_names = None
        self.metrics = {}

    # ----------------------------------------------------------
    # Data Loading
    # ----------------------------------------------------------

    def load_training_data(self):
        """
        Load mature loans from PostgreSQL with binary default labels.

        Returns:
            pd.DataFrame: Features + target column 'is_default'.
        """
        print("Loading mature loans from database...")

        status_list = "', '".join(ALL_MATURE)
        query = "SELECT * FROM loans_master WHERE loan_status IN ('{}')".format(
            status_list)

        df = pd.read_sql(query, get_engine())
        print("   -> Loaded {:,} mature loans.".format(len(df)))

        # Binary label
        df['is_default'] = df['loan_status'].isin(DEFAULTED_STATUSES).astype(int)

        n_default = df['is_default'].sum()
        n_paid = len(df) - n_default
        default_rate = n_default / len(df)

        print("   -> Defaulted: {:,} ({:.2%})".format(n_default, default_rate))
        print("   -> Fully Paid: {:,} ({:.2%})".format(n_paid, 1 - default_rate))

        return df

    # ----------------------------------------------------------
    # Feature Engineering
    # ----------------------------------------------------------

    def prepare_features(self, df, fit_encoders=True):
        """
        Select, encode, and scale features.

        Args:
            df: Raw DataFrame with feature columns.
            fit_encoders: If True, fit new encoders (training).
                          If False, use existing encoders (inference).

        Returns:
            pd.DataFrame: Processed feature matrix.
        """
        available_features = [c for c in FEATURE_COLS if c in df.columns]
        print("   -> Using {}/{} features.".format(
            len(available_features), len(FEATURE_COLS)))

        X = df[available_features].copy()

        # --- Derived features ---
        X['fico_midpoint'] = (X['fico_range_low'] + X['fico_range_high']) / 2.0

        X['loan_to_income'] = np.where(
            X['annual_inc'] > 0,
            X['loan_amnt'] / X['annual_inc'],
            0.0
        )

        # Income burden: monthly obligations relative to income
        if 'installment' in X.columns:
            X['monthly_burden'] = np.where(
                X['annual_inc'] > 0,
                (X['installment'] * 12) / X['annual_inc'],
                0.0
            )
        else:
            X['monthly_burden'] = np.where(
                X['annual_inc'] > 0,
                (X['int_rate'] / 100 * X['loan_amnt']) / X['annual_inc'],
                0.0
            )

        # --- Encode categorical columns ---
        for col in CATEGORICAL_COLS:
            if col not in X.columns:
                continue

            if fit_encoders:
                le = LabelEncoder()
                X[col] = X[col].astype(str).fillna('Unknown')
                le.fit(X[col])
                self.encoders[col] = le
                print("   -> Fitted encoder for '{}': {} classes".format(
                    col, len(le.classes_)))
            else:
                le = self.encoders.get(col)
                if le is None:
                    print("   -> WARNING: No encoder found for '{}', skipping.".format(col))
                    continue
                X[col] = X[col].astype(str).fillna('Unknown')
                X[col] = X[col].apply(lambda x: x if x in le.classes_ else 'Unknown')

            X[col] = le.transform(X[col])

        # --- Fill missing numeric values ---
        X = X.fillna(0)

        # --- Scale ---
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if fit_encoders:
            self.scaler = StandardScaler()
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        else:
            if self.scaler is not None:
                X[numeric_cols] = self.scaler.transform(X[numeric_cols])

        self.feature_names = X.columns.tolist()
        return X

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    def train(self, test_size=0.2, random_state=42):
        """
        Full training pipeline:
        1. Load mature loans
        2. Engineer features
        3. Train/test split
        4. Train HistGradientBoostingClassifier
        5. Evaluate metrics (AUC-ROC, precision, recall)
        6. Feature importance
        7. Save model and dashboard artefacts

        Args:
            test_size: Fraction of data for holdout test set.
            random_state: Random seed for reproducibility.
        """
        print("=" * 60)
        print("TRAINING PD MODEL")
        print("=" * 60)

        # Step 1: Load
        df = self.load_training_data()

        # Step 2: Features
        print("\nPreparing features...")
        X = self.prepare_features(df, fit_encoders=True)
        y = df['is_default']

        # Step 3: Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print("   -> Train: {:,} | Test: {:,}".format(len(X_train), len(X_test)))
        print("   -> Train default rate: {:.4f}".format(y_train.mean()))
        print("   -> Test  default rate: {:.4f}".format(y_test.mean()))

        # Step 4: Train
        print("\nTraining HistGradientBoostingClassifier...")
        self.model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=5,
            learning_rate=0.1,
            max_bins=255,
            min_samples_leaf=50,
            l2_regularization=0.0,
            random_state=random_state
        )
        self.model.fit(X_train, y_train)
        print("   -> Training complete.")

        # Step 5: Evaluate
        print("\nEvaluating on test set...")
        y_prob = self.model.predict_proba(X_test)[:, 1]  # Probability of class 1 (default)
        y_pred = (y_prob >= 0.5).astype(int)

        # Core metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Confusion matrix (2x2 list, JSON-serialisable)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Classification report as plain text
        clf_report = classification_report(
            y_test, y_pred,
            target_names=['Non-Default', 'Default'],
            digits=4
        )

        self.metrics = {
            'auc_roc': round(float(auc), 4),
            'accuracy': round(float(accuracy), 4),
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
            'f1_score': round(float(f1), 4),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': len(self.feature_names),
            'default_rate_train': round(float(y_train.mean()), 4),
            'default_rate_test': round(float(y_test.mean()), 4)
        }

        print("\n   PD Model Metrics:")
        print("     AUC-ROC:   {:.4f}".format(auc))
        print("     Accuracy:  {:.4f}".format(accuracy))
        print("     Precision: {:.4f}".format(precision))
        print("     Recall:    {:.4f}".format(recall))
        print("     F1 Score:  {:.4f}".format(f1))

        print("\n   Confusion Matrix (threshold=0.5):")
        print("                  Predicted")
        print("               Non-Def   Default")
        print("   Actual Non-Def  {:>7,}  {:>7,}".format(cm[0][0], cm[0][1]))
        print("   Actual Default  {:>7,}  {:>7,}".format(cm[1][0], cm[1][1]))

        print("\n   Classification Report:")
        print(clf_report)

        # Step 6: Feature Importance
        self._print_feature_importance()

        # Step 7: Save
        self.save_model()

        # Step 8: Export dashboard artefacts
        self._export_dashboard(
            accuracy=accuracy, auc=auc, precision=precision,
            recall=recall, f1=f1, cm=cm, clf_report=clf_report,
            y_test=y_test, y_pred=y_pred
        )

        print("\n" + "=" * 60)
        print("PD MODEL TRAINING COMPLETE")
        print("=" * 60)

        return self.metrics

    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------

    def predict_proba(self, df):
        """
        Predict probability of default for new loans.

        Args:
            df: DataFrame with the same feature columns as training data.

        Returns:
            np.ndarray: PD values in [0, 1].
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = self.prepare_features(df, fit_encoders=False)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df, threshold=0.5):
        """
        Predict default class for new loans.

        Args:
            df: DataFrame with the same feature columns.
            threshold: Classification threshold.

        Returns:
            np.ndarray: Binary predictions (0 or 1).
        """
        pd_values = self.predict_proba(df)
        return (pd_values >= threshold).astype(int)

    # ----------------------------------------------------------
    # Model Persistence
    # ----------------------------------------------------------

    def save_model(self):
        """Save model, scaler, and encoders to disk."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(self.encoders, ENCODERS_PATH)

        print("\n   Model saved to: {}".format(MODEL_PATH))
        print("   Scaler saved to: {}".format(SCALER_PATH))
        print("   Encoders saved to: {}".format(ENCODERS_PATH))

    def load_model(self):
        """Load a previously saved model, scaler, and encoders."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model not found at {}".format(MODEL_PATH))

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.encoders = joblib.load(ENCODERS_PATH)

        print("Model loaded from: {}".format(MODEL_PATH))
        print("   Features: {}".format(self.model.n_features_in_))
        return self

    # ----------------------------------------------------------
    # Dashboard Export
    # ----------------------------------------------------------

    def _export_dashboard(self, accuracy, auc, precision, recall, f1,
                          cm, clf_report, y_test, y_pred):
        """
        Export model metrics and feature importance for the
        Streamlit visualization dashboard.

        Files produced:
            output/models/pd_model_metrics.json
            output/models/pd_feature_importance.csv
        """
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # --- JSON metrics ---
        metrics_json = {
            "auc_roc": round(float(auc), 4),
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "classification_report": clf_report,
            "confusion_matrix": cm,
            "n_train": int(len(y_test) + y_test.sum()),
            "n_test": int(len(y_test)),
        }

        with open(METRICS_PATH, "w") as f:
            json.dump(metrics_json, f, indent=2)
        print("\n   Dashboard metrics saved to: {}".format(METRICS_PATH))

        # --- Feature importance CSV ---
        if hasattr(self.model, "feature_importances_"):
            fi_df = pd.DataFrame({
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }).sort_values("importance", ascending=False).reset_index(drop=True)

            fi_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
            print("   Feature importance saved to: {}".format(FEATURE_IMPORTANCE_PATH))

    # ----------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------

    def _print_feature_importance(self, top_n=15):
        """Print the top N most important features."""
        if self.model is None:
            return

        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n   Top {} Feature Importances:".format(top_n))
        print("   " + "-" * 50)
        for _, row in importances.head(top_n).iterrows():
            bar = "#" * int(row['importance'] * 200)
            print("   {:<30} {:.4f}  {}".format(
                row['feature'], row['importance'], bar))

    def get_feature_importance(self):
        """Return feature importance as a DataFrame."""
        if self.model is None:
            raise RuntimeError("Model not trained yet.")

        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).reset_index(drop=True)

    def summary_by_grade(self, df=None):
        """
        Compare actual vs predicted default rates by grade.

        Args:
            df: Raw DataFrame. If None, reloads from database.

        Returns:
            pd.DataFrame: Summary by grade.
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet.")

        if df is None:
            df = self.load_training_data()

        df['predicted_pd'] = self.predict_proba(df)

        summary = df.groupby('grade').agg(
            count=('is_default', 'size'),
            actual_default_rate=('is_default', 'mean'),
            predicted_avg_pd=('predicted_pd', 'mean'),
            predicted_median_pd=('predicted_pd', 'median')
        ).round(4)

        summary['error'] = (summary['actual_default_rate'] - summary['predicted_avg_pd']).abs().round(4)
        summary = summary.sort_index()

        print("\n   PD Summary by Grade (Actual vs Predicted):")
        print(summary.to_string())

        return summary

    def summary_by_segment(self, df=None, segment_col='term'):
        """
        Compare actual vs predicted default rates by an arbitrary segment.

        Args:
            df: Raw DataFrame. If None, reloads from database.
            segment_col: Column to group by.

        Returns:
            pd.DataFrame: Summary by segment.
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet.")

        if df is None:
            df = self.load_training_data()

        if segment_col not in df.columns:
            print("   Column '{}' not found.".format(segment_col))
            return None

        df['predicted_pd'] = self.predict_proba(df)

        summary = df.groupby(segment_col).agg(
            count=('is_default', 'size'),
            actual_default_rate=('is_default', 'mean'),
            predicted_avg_pd=('predicted_pd', 'mean')
        ).round(4)

        summary['error'] = (summary['actual_default_rate'] - summary['predicted_avg_pd']).abs().round(4)

        print("\n   PD Summary by {}:".format(segment_col))
        print(summary.to_string())

        return summary


# ==================================================================
# Standalone Execution
# ==================================================================
if __name__ == "__main__":
    try:
        model = PDModel()
        model.train()
        model.summary_by_grade()
        model.summary_by_segment(segment_col='term')
    except Exception as e:
        print("ERROR: {}".format(e))
        import traceback
        traceback.print_exc()
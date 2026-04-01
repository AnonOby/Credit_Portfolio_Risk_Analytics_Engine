"""
PD (Probability of Default) Prediction Model

Trains a binary classifier to predict the probability of loan default.
Uses mature loans only (Fully Paid or Charged Off / Default) to ensure
ground truth labels are available.

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
import sys
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, classification_report,
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


class PDModel:
    """
    Probability of Default prediction model.

    Uses GradientBoostingClassifier to predict the likelihood
    of a loan defaulting. Outputs calibrated PD scores in [0, 1].
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
        query = f"""
            SELECT *
            FROM loans_master
            WHERE loan_status IN ('{status_list}')
        """

        df = pd.read_sql(query, get_engine())
        print(f"   -> Loaded {len(df):,} mature loans.")

        # Binary label
        df['is_default'] = df['loan_status'].isin(DEFAULTED_STATUSES).astype(int)

        n_default = df['is_default'].sum()
        n_paid = len(df) - n_default
        default_rate = n_default / len(df)

        print(f"   -> Defaulted: {n_default:,} ({default_rate:.2%})")
        print(f"   -> Fully Paid: {n_paid:,} ({1 - default_rate:.2%})")

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
        print(f"   -> Using {len(available_features)}/{len(FEATURE_COLS)} features.")

        X = df[available_features].copy()

        # --- Derived features ---
        X['fico_midpoint'] = (X['fico_range_low'] + X['fico_range_high']) / 2.0

        X['loan_to_income'] = np.where(
            X['annual_inc'] > 0,
            X['loan_amnt'] / X['annual_inc'],
            0.0
        )

        # Income burden: monthly obligations relative to income
        X['monthly_burden'] = np.where(
            X['annual_inc'] > 0,
            (X['installment'] * 12) / X['annual_inc'],
            0.0
        ) if 'installment' in X.columns else np.where(
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
                print(f"   -> Fitted encoder for '{col}': {len(le.classes_)} classes")
            else:
                le = self.encoders.get(col)
                if le is None:
                    print(f"   -> WARNING: No encoder found for '{col}', skipping.")
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
        4. Train GradientBoostingClassifier
        5. Evaluate metrics (AUC-ROC, precision, recall)
        6. Feature importance

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
        print(f"   -> Train: {len(X_train):,} | Test: {len(X_test):,}")
        print(f"   -> Train default rate: {y_train.mean():.4f}")
        print(f"   -> Test  default rate: {y_test.mean():.4f}")

        # Step 4: Train
        print("\nTraining GradientBoostingClassifier...")
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=50,
            random_state=random_state
        )
        self.model.fit(X_train, y_train)
        print("   -> Training complete.")

        # Step 5: Evaluate
        print("\nEvaluating on test set...")
        y_prob = self.model.predict_proba(X_test)[:, 1]  # Probability of class 1 (default)
        y_pred = (y_prob >= 0.5).astype(int)

        # Core metrics
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        self.metrics = {
            'auc_roc': round(auc, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': len(self.feature_names),
            'default_rate_train': round(y_train.mean(), 4),
            'default_rate_test': round(y_test.mean(), 4)
        }

        print(f"\n   PD Model Metrics:")
        print(f"     AUC-ROC:   {auc:.4f}")
        print(f"     Precision: {precision:.4f}")
        print(f"     Recall:    {recall:.4f}")
        print(f"     F1 Score:  {f1:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n   Confusion Matrix (threshold=0.5):")
        print(f"                  Predicted")
        print(f"               Non-Def   Default")
        print(f"   Actual Non-Def  {cm[0][0]:>7,}  {cm[0][1]:>7,}")
        print(f"   Actual Default  {cm[1][0]:>7,}  {cm[1][1]:>7,}")

        # Classification report
        print(f"\n   Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=['Non-Default', 'Default'],
            digits=4
        ))

        # Step 6: Feature Importance
        self._print_feature_importance()

        # Step 7: Save
        self.save_model()

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

        print(f"\n   Model saved to: {MODEL_PATH}")
        print(f"   Scaler saved to: {SCALER_PATH}")
        print(f"   Encoders saved to: {ENCODERS_PATH}")

    def load_model(self):
        """Load a previously saved model, scaler, and encoders."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.encoders = joblib.load(ENCODERS_PATH)

        print(f"Model loaded from: {MODEL_PATH}")
        print(f"   Features: {self.model.n_features_in_}")
        return self

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

        print(f"\n   Top {top_n} Feature Importances:")
        print("   " + "-" * 50)
        for _, row in importances.head(top_n).iterrows():
            bar = '#' * int(row['importance'] * 200)
            print(f"   {row['feature']:<30} {row['importance']:.4f}  {bar}")

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
            print(f"   Column '{segment_col}' not found.")
            return None

        df['predicted_pd'] = self.predict_proba(df)

        summary = df.groupby(segment_col).agg(
            count=('is_default', 'size'),
            actual_default_rate=('is_default', 'mean'),
            predicted_avg_pd=('predicted_pd', 'mean')
        ).round(4)

        summary['error'] = (summary['actual_default_rate'] - summary['predicted_avg_pd']).abs().round(4)

        print(f"\n   PD Summary by {segment_col}:")
        print(summary.to_string())

        return summary


# ==================================================================
# Standalone Execution
# ==================================================================
if __name__ == "__main__":
    from sklearn.ensemble import GradientBoostingClassifier

    try:
        model = PDModel()
        model.train()
        model.summary_by_grade()
        model.summary_by_segment(segment_col='term')
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
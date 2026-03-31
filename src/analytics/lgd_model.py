"""
LGD (Loss Given Default) Prediction Model

Trains a regression model to predict recovery rates for defaulted loans,
then derives LGD = 1 - recovery_rate.

Target:
    recovery_rate = total_pymnt / funded_amnt
    LGD = 1 - recovery_rate

Features:
    - Grade / sub-grade, term, interest rate, loan amount
    - FICO score, DTI, annual income, employment length
    - Credit history, delinquencies, revolving utilization
    - Purpose, home ownership (encoded)
    - Census economic indicators (income, growth rates)
"""

import pandas as pd
import numpy as np
import sys
import os
import sklearn
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database.connection import get_engine
import config


# ==================================================================
# Configuration
# ==================================================================

DEFAULTED_STATUSES = [
    'Charged Off',
    'Default',
    'Does not meet the credit policy. Status:Charged Off'
]

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

# Model output path
MODEL_DIR = config.BASE_DIR / 'output' / 'models'
MODEL_PATH = MODEL_DIR / 'lgd_model.joblib'
SCALER_PATH = MODEL_DIR / 'lgd_scaler.joblib'
ENCODERS_PATH = MODEL_DIR / 'lgd_encoders.joblib'


class LGDModel:
    """
    Loss Given Default prediction model.

    Predicts recovery rate using Gradient Boosting Regressor,
    then converts to LGD:  LGD = 1 - predicted_recovery_rate.
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
        Load defaulted loans from PostgreSQL and compute LGD target.

        Returns:
            pd.DataFrame: Features + target (recovery_rate) + raw LGD.
        """
        print("Loading defaulted loans from database...")

        status_list = "', '".join(DEFAULTED_STATUSES)
        query = f"""
            SELECT *
            FROM loans_master
            WHERE loan_status IN ('{status_list}')
              AND funded_amnt > 0
        """

        df = pd.read_sql(query, get_engine())
        print(f"   -> Loaded {len(df):,} defaulted loans.")

        # Compute recovery rate and LGD
        df['recovery_rate'] = df['total_pymnt'] / df['funded_amnt']
        df['lgd'] = 1 - df['recovery_rate']

        print(f"   -> Recovery rate: mean={df['recovery_rate'].mean():.4f}, "
              f"median={df['recovery_rate'].median():.4f}")
        print(f"   -> LGD:            mean={df['lgd'].mean():.4f}, "
              f"median={df['lgd'].median():.4f}")

        return df

    # ----------------------------------------------------------
    # Feature Engineering
    # ----------------------------------------------------------

    def prepare_features(self, df, fit_encoders=True):
        """
        Select, encode, and scale features for model training or inference.

        Args:
            df: Raw DataFrame with feature columns.
            fit_encoders: If True, fit new encoders (training).
                          If False, use existing encoders (inference).

        Returns:
            pd.DataFrame: Processed feature matrix.
        """
        # Filter to available columns only
        available_features = [c for c in FEATURE_COLS if c in df.columns]
        print(f"   -> Using {len(available_features)}/{len(FEATURE_COLS)} features.")

        X = df[available_features].copy()

        # --- Derived features ---
        X['fico_midpoint'] = (X['fico_range_low'] + X['fico_range_high']) / 2.0

        # Loan-to-income ratio (handle zero income)
        X['loan_to_income'] = np.where(
            X['annual_inc'] > 0,
            X['loan_amnt'] / X['annual_inc'],
            0.0
        )

        # Interest burden: installment relative to income
        X['credit_utilization_combined'] = np.where(
            X['annual_inc'] > 0,
            X['revol_bal'] / X['annual_inc'],
            0.0
        )

        # --- Encode categorical columns ---
        for col in CATEGORICAL_COLS:
            if col not in X.columns:
                continue

            if fit_encoders:
                le = LabelEncoder()
                # Handle unseen labels safely
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
                # Map unseen labels to 'Unknown'
                X[col] = X[col].apply(lambda x: x if x in le.classes_ else 'Unknown')

            X[col] = le.transform(X[col])

        # --- Fill missing numeric values with 0 ---
        X = X.fillna(0)

        # --- Scale numeric features ---
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
        1. Load data
        2. Engineer features
        3. Train/test split
        4. Train GradientBoostingRegressor
        5. Evaluate and store metrics

        Args:
            test_size: Fraction of data for holdout test set.
            random_state: Random seed for reproducibility.
        """
        print("=" * 60)
        print("TRAINING LGD MODEL")
        print("=" * 60)

        # Step 1: Load
        df = self.load_training_data()

        # Step 2: Features
        print("\nPreparing features...")
        X = self.prepare_features(df, fit_encoders=True)
        y = df['recovery_rate']  # Predict recovery rate, derive LGD later

        # Step 3: Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"   -> Train: {len(X_train):,} | Test: {len(X_test):,}")

        # Step 4: Train
        print("\nTraining GradientBoostingRegressor...")
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=50,
            loss='huber',         # Robust to outliers in recovery rates
            random_state=random_state
        )
        self.model.fit(X_train, y_train)
        print("   -> Training complete.")

        # Step 5: Evaluate
        print("\nEvaluating on test set...")
        y_pred = self.model.predict(X_test)

        # Clamp predictions to [0, 1]
        y_pred_clamped = np.clip(y_pred, 0.0, 1.0)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred_clamped))
        mae = mean_absolute_error(y_test, y_pred_clamped)
        r2 = r2_score(y_test, y_pred_clamped)

        # Convert to LGD metrics
        y_test_lgd = 1 - y_test
        y_pred_lgd = 1 - y_pred_clamped

        lgd_rmse = np.sqrt(mean_squared_error(y_test_lgd, y_pred_lgd))
        lgd_mae = mean_absolute_error(y_test_lgd, y_pred_lgd)

        self.metrics = {
            'recovery_rmse': round(rmse, 4),
            'recovery_mae': round(mae, 4),
            'recovery_r2': round(r2, 4),
            'lgd_rmse': round(lgd_rmse, 4),
            'lgd_mae': round(lgd_mae, 4),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': len(self.feature_names)
        }

        print(f"\n   Recovery Rate Metrics:")
        print(f"     RMSE: {rmse:.4f}")
        print(f"     MAE:  {mae:.4f}")
        print(f"     R2:   {r2:.4f}")
        print(f"\n   LGD Metrics:")
        print(f"     RMSE: {lgd_rmse:.4f}")
        print(f"     MAE:  {lgd_mae:.4f}")

        # Step 6: Feature Importance
        self._print_feature_importance()

        # Step 7: Save
        self.save_model()
        print("\n" + "=" * 60)
        print("LGD MODEL TRAINING COMPLETE")
        print("=" * 60)

        return self.metrics

    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------

    def predict(self, df):
        """
        Predict LGD for new loans.

        Args:
            df: DataFrame with the same feature columns as training data.

        Returns:
            np.ndarray: Predicted LGD values (clipped to [0, 1]).
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = self.prepare_features(df, fit_encoders=False)
        recovery_pred = self.model.predict(X)
        lgd_pred = 1 - np.clip(recovery_pred, 0.0, 1.0)
        return lgd_pred

    def predict_recovery(self, df):
        """
        Predict recovery rate for new loans.

        Args:
            df: DataFrame with the same feature columns as training data.

        Returns:
            np.ndarray: Predicted recovery rates (clipped to [0, 1]).
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = self.prepare_features(df, fit_encoders=False)
        recovery_pred = np.clip(self.model.predict(X), 0.0, 1.0)
        return recovery_pred

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
        print("   " + "-" * 45)
        for i, row in importances.head(top_n).iterrows():
            bar = '🦚' * int(row['importance'] * 200)
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
        Compare actual vs predicted LGD by grade.

        Args:
            df: Raw DataFrame with features + total_pymnt + funded_amnt.
                If None, reloads from database.

        Returns:
            pd.DataFrame: Summary by grade with actual and predicted LGD.
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet.")

        if df is None:
            df = self.load_training_data()

        df['predicted_lgd'] = self.predict(df)
        df['actual_lgd'] = df['lgd']

        summary = df.groupby('grade').agg(
            count=('lgd', 'size'),
            actual_lgd_mean=('actual_lgd', 'mean'),
            predicted_lgd_mean=('predicted_lgd', 'mean'),
            actual_lgd_median=('actual_lgd', 'median'),
            predicted_lgd_median=('predicted_lgd', 'median')
        ).round(4)

        summary['error'] = (summary['actual_lgd_mean'] - summary['predicted_lgd_mean']).abs().round(4)

        print("\n   LGD Summary by Grade (Actual vs Predicted):")
        print(summary.to_string())

        return summary


# ==================================================================
# Standalone Execution
# ==================================================================
if __name__ == "__main__":
    try:
        model = LGDModel()
        model.train()
        model.summary_by_grade()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
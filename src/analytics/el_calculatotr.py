"""
Expected Loss (EL) Calculator

Combines PD and LGD predictions to compute Expected Loss at both
loan and portfolio level.

Core formula:
    EL = PD x LGD x EAD

Where:
    PD  = Probability of Default (from PD model or historical)
    LGD = Loss Given Default (from LGD model or historical)
    EAD = Exposure at Default (= funded_amnt for installment loans)

The calculator supports two modes:
    1. Model-based: Uses trained PD and LGD models for predictions
    2. Historical: Uses SQL-computed historical PD and LGD per grade

Outputs:
    - Per-loan EL estimates
    - Portfolio EL aggregated by grade, segment, and purpose
    - Comparison: model-based vs historical EL
    - CSV exports for downstream reporting
"""

import pandas as pd
import sys
import os

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

ALL_MATURE = ['Fully Paid'] + DEFAULTED_STATUSES

OUTPUT_DIR = config.BASE_DIR / 'output'
LOAN_EL_PATH = OUTPUT_DIR / 'loan_level_el.csv'
PORTFOLIO_EL_PATH = OUTPUT_DIR / 'portfolio_el_summary.csv'
COMPARISON_PATH = OUTPUT_DIR / 'el_model_vs_historical.csv'


class ExpectedLossCalculator:
    """
    Computes Expected Loss at loan and portfolio level.

    Supports both model-based predictions and historical estimates,
    allowing comparison between the two approaches.
    """

    def __init__(self):
        self.loan_el = None
        self.portfolio_summary = None
        self.comparison = None

    # ----------------------------------------------------------
    # Data Loading
    # ----------------------------------------------------------

    def load_historical_rates(self):
        """
        Load historical PD and LGD per grade from SQL analytics.

        Returns:
            dict: Mapping of grade -> {pd, lgd}
        """
        print("Loading historical PD and LGD from database...")

        pd_query = """
            WITH mature_loans AS (
                SELECT * FROM loans_master
                WHERE loan_status IN (
                    'Fully Paid', 'Charged Off', 'Default',
                    'Does not meet the credit policy. Status:Fully Paid',
                    'Does not meet the credit policy. Status:Charged Off'
                )
            )
            SELECT
                grade,
                SUM(CASE WHEN loan_status IN ('Charged Off', 'Default',
                    'Does not meet the credit policy. Status:Charged Off')
                    THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS historical_pd
            FROM mature_loans
            GROUP BY grade
        """
        pd_df = pd.read_sql(pd_query, get_engine())

        lgd_query = """
            SELECT
                grade,
                AVG(1 - (total_pymnt / funded_amnt)) AS historical_lgd
            FROM loans_master
            WHERE loan_status IN ('Charged Off', 'Default',
                'Does not meet the credit policy. Status:Charged Off')
            AND funded_amnt > 0
            GROUP BY grade
        """
        lgd_df = pd.read_sql(lgd_query, get_engine())

        rates = {}
        for _, row in pd_df.iterrows():
            g = row['grade']
            rates[g] = {'pd': row['historical_pd']}

        for _, row in lgd_df.iterrows():
            g = row['grade']
            if g not in rates:
                rates[g] = {}
            rates[g]['lgd'] = row['historical_lgd']

        for g, vals in rates.items():
            print(f"   -> Grade {g}: PD={vals.get('pd', 'N/A'):.4f}, LGD={vals.get('lgd', 'N/A'):.4f}")

        return rates

    def load_portfolio(self):
        """
        Load all loans from database with all feature columns.
        All columns needed by PD and LGD models must be present.

        Returns:
            pd.DataFrame: All loans.
        """
        print("Loading portfolio from database...")

        query = "SELECT * FROM loans_master"

        df = pd.read_sql(query, get_engine())
        print(f"   -> Loaded {len(df):,} loans ({len(df.columns)} columns).")
        return df

    # ----------------------------------------------------------
    # EL Computation — Historical
    # ----------------------------------------------------------

    def compute_el_historical(self, df, rates):
        """
        Compute EL using historical PD and LGD per grade.

        EL_i = PD_grade(i) * LGD_grade(i) * EAD_i

        Args:
            df: Portfolio DataFrame with grade and ead columns.
            rates: Dict of grade -> {pd, lgd}.

        Returns:
            pd.DataFrame: DataFrame with EL columns added.
        """
        print("\nComputing historical EL...")

        df = df.copy()
        df['pd_historical'] = df['grade'].map(lambda g: rates.get(g, {}).get('pd', 0.20))
        df['lgd_historical'] = df['grade'].map(lambda g: rates.get(g, {}).get('lgd', 0.46))
        df['el_historical'] = df['pd_historical'] * df['lgd_historical'] * df['ead']

        total_el = df['el_historical'].sum()
        total_ead = df['ead'].sum()

        print(f"   -> Total EL (historical): ${total_el:,.0f}")
        print(f"   -> Portfolio EL rate:    {total_el / total_ead:.4%}")
        print(f"   -> Total EAD:            ${total_ead:,.0f}")

        return df

    # ----------------------------------------------------------
    # EL Computation — Model-based
    # ----------------------------------------------------------

    def compute_el_model(self, df):
        """
        Compute EL using trained PD and LGD models.

        Requires pre-trained model files in output/models/.

        Args:
            df: Portfolio DataFrame.

        Returns:
            pd.DataFrame: DataFrame with model PD, LGD, EL columns.
        """
        print("\nComputing model-based EL...")
        print("   -> Loading PD model...")
        try:
            from src.analytics.pd_model import PDModel
            pd_model = PDModel().load_model()
            print("   -> PD model loaded successfully.")
        except Exception as e:
            print(f"   -> WARNING: Could not load PD model: {e}")
            print("   -> Falling back to historical PD.")
            pd_model = None

        print("   -> Loading LGD model...")
        try:
            from src.analytics.lgd_model import LGDModel
            lgd_model = LGDModel().load_model()
            print("   -> LGD model loaded successfully.")
        except Exception as e:
            print(f"   -> WARNING: Could not load LGD model: {e}")
            print("   -> Falling back to historical LGD.")
            lgd_model = None

        df = df.copy()

        # PD predictions (sample for speed if portfolio is large)
        if pd_model is not None:
            print("   -> Predicting PD for all loans...")
            df['pd_model'] = pd_model.predict_proba(df)
            print(f"   -> PD range: {df['pd_model'].min():.4f} to {df['pd_model'].max():.4f}")
        else:
            rates = self.load_historical_rates()
            df['pd_model'] = df['grade'].map(lambda g: rates.get(g, {}).get('pd', 0.20))

        # LGD predictions
        if lgd_model is not None:
            print("   -> Predicting LGD for all loans...")
            df['lgd_model'] = lgd_model.predict(df)
            print(f"   -> LGD range: {df['lgd_model'].min():.4f} to {df['lgd_model'].max():.4f}")
        else:
            rates = self.load_historical_rates()
            df['lgd_model'] = df['grade'].map(lambda g: rates.get(g, {}).get('lgd', 0.46))

        # Compute EL
        df['el_model'] = df['pd_model'] * df['lgd_model'] * df['ead']

        total_el = df['el_model'].sum()
        total_ead = df['ead'].sum()

        print(f"   -> Total EL (model):      ${total_el:,.0f}")
        print(f"   -> Portfolio EL rate:     {total_el / total_ead:.4%}")

        return df

    # ----------------------------------------------------------
    # Portfolio Aggregation
    # ----------------------------------------------------------

    def aggregate_portfolio(self, df):
        """
        Aggregate EL metrics at portfolio level by various segments.

        Args:
            df: DataFrame with EL columns.

        Returns:
            pd.DataFrame: Portfolio summary by grade and segment.
        """
        print("\nAggregating portfolio EL metrics...")

        # --- By Grade ---
        grade_summary = df.groupby('grade').agg(
            n_loans=('id', 'count'),
            total_ead=('ead', 'sum'),
            avg_pd=('pd_model', 'mean') if 'pd_model' in df.columns else ('pd_historical', 'mean'),
            avg_lgd=('lgd_model', 'mean') if 'lgd_model' in df.columns else ('lgd_historical', 'mean'),
            total_el=('el_historical', 'sum'),
            el_rate=('el_historical', lambda x: x.sum() / df.loc[x.index, 'ead'].sum())
        ).round(4)

        if 'el_model' in df.columns:
            grade_summary['total_el_model'] = df.groupby('grade')['el_model'].sum()
            grade_summary['el_rate_model'] = grade_summary['total_el_model'] / grade_summary['total_ead']
            grade_summary['el_diff'] = (grade_summary['total_el_model'] - grade_summary['total_el']).abs()

        grade_summary = grade_summary.sort_index()

        print(f"\n   {'Grade':<6} {'Loans':>10} {'EAD':>16} {'PD':>8} {'LGD':>8} {'EL Rate':>10} {'Total EL':>16}")
        print(f"   {'-' * 80}")
        for idx, row in grade_summary.iterrows():
            el_col = 'el_rate_model' if 'el_rate_model' in row.index else 'el_rate'
            el_total_col = 'total_el_model' if 'total_el_model' in row.index else 'total_el'
            print(f"   {idx:<6} {row['n_loans']:>10,} ${row['total_ead']:>14,.0f} "
                  f"{row['avg_pd']:>7.4f} {row['avg_lgd']:>7.4f} "
                  f"{row[el_col]:>9.4%} ${row[el_total_col]:>14,.0f}")

        self.portfolio_summary = grade_summary
        return grade_summary

    def aggregate_by_segment(self, df, segment_col):
        """
        Aggregate EL by an arbitrary segment column.

        Args:
            df: DataFrame with EL columns.
            segment_col: Column to group by.

        Returns:
            pd.DataFrame: Summary by segment.
        """
        el_col = 'el_model' if 'el_model' in df.columns else 'el_historical'

        summary = df.groupby(segment_col).agg(
            n_loans=('id', 'count'),
            total_ead=('ead', 'sum'),
            total_el=(el_col, 'sum')
        ).assign(el_rate=lambda x: x['total_el'] / x['total_ead'])

        summary = summary.sort_values('total_el', ascending=False)
        return summary

    # ----------------------------------------------------------
    # Comparison: Model vs Historical
    # ----------------------------------------------------------

    def compare_model_historical(self, df):
        """
        Compare model-based EL with historical EL.

        Args:
            df: DataFrame with both el_historical and el_model columns.

        Returns:
            pd.DataFrame: Comparison summary.
        """
        if 'el_model' not in df.columns or 'el_historical' not in df.columns:
            print("   -> Both model and historical EL required for comparison.")
            return None

        print("\nComparing model-based vs historical EL...")

        comparison = df.groupby('grade').agg(
            n_loans=('id', 'count'),
            total_ead=('ead', 'sum'),
            total_el_historical=('el_historical', 'sum'),
            total_el_model=('el_model', 'sum')
        ).round(4)

        comparison['el_rate_historical'] = comparison['total_el_historical'] / comparison['total_ead']
        comparison['el_rate_model'] = comparison['total_el_model'] / comparison['total_ead']
        comparison['rate_diff'] = (comparison['el_rate_model'] - comparison['el_rate_historical']).abs()
        comparison['rate_diff_pct'] = (comparison['rate_diff'] / comparison['el_rate_historical'] * 100).round(2)

        # Totals row
        totals = pd.DataFrame({
            'n_loans': [comparison['n_loans'].sum()],
            'total_ead': [comparison['total_ead'].sum()],
            'total_el_historical': [comparison['total_el_historical'].sum()],
            'total_el_model': [comparison['total_el_model'].sum()],
            'el_rate_historical': [comparison['total_el_historical'].sum() / comparison['total_ead'].sum()],
            'el_rate_model': [comparison['total_el_model'].sum() / comparison['total_ead'].sum()],
            'rate_diff': [abs(comparison['total_el_model'].sum() / comparison['total_ead'].sum() -
                           comparison['total_el_historical'].sum() / comparison['total_ead'].sum())]
        }, index=['TOTAL'])

        comparison = pd.concat([comparison, totals])

        print(f"\n   {'Grade':<8} {'Loans':>10} {'EL Rate (Hist)':>14} {'EL Rate (Model)':>15} {'Diff':>10}")
        print(f"   {'-' * 60}")
        for idx, row in comparison.iterrows():
            marker = ' <--' if idx == 'TOTAL' else ''
            print(f"   {idx:<8} {row['n_loans']:>10,} {row['el_rate_historical']:>13.4%} "
                  f"{row['el_rate_model']:>14.4%} {row['rate_diff']:>9.4%}{marker}")

        self.comparison = comparison
        return comparison

    # ----------------------------------------------------------
    # Export
    # ----------------------------------------------------------

    def export_results(self, df):
        """
        Export EL results to CSV files.

        Args:
            df: DataFrame with EL columns.
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Loan-level EL (sample to avoid huge files)
        el_col = 'el_model' if 'el_model' in df.columns else 'el_historical'
        pd_col = 'pd_model' if 'pd_model' in df.columns else 'pd_historical'
        lgd_col = 'lgd_model' if 'lgd_model' in df.columns else 'lgd_historical'

        loan_export = df[['id', 'grade', 'sub_grade', 'ead', pd_col, lgd_col, el_col]].copy()
        loan_export.columns = ['id', 'grade', 'sub_grade', 'ead', 'pd', 'lgd', 'el']
        loan_export.to_csv(LOAN_EL_PATH, index=False)
        print(f"   -> Loan-level EL saved to: {LOAN_EL_PATH} ({len(loan_export):,} rows)")

        # Portfolio summary
        if self.portfolio_summary is not None:
            self.portfolio_summary.to_csv(PORTFOLIO_EL_PATH)
            print(f"   -> Portfolio summary saved to: {PORTFOLIO_EL_PATH}")

        # Model vs Historical comparison
        if self.comparison is not None:
            self.comparison.to_csv(COMPARISON_PATH)
            print(f"   -> EL comparison saved to: {COMPARISON_PATH}")

    # ----------------------------------------------------------
    # Full Pipeline
    # ----------------------------------------------------------

    def run(self, use_models=True):
        """
        Execute the full EL calculation pipeline.

        Args:
            use_models: If True, use trained PD/LGD models.
                        If False, use historical rates only.

        Returns:
            tuple: (DataFrame with EL, portfolio summary, comparison)
        """
        print("=" * 60)
        print("EXPECTED LOSS CALCULATOR")
        print("=" * 60)

        # Step 1: Load portfolio
        df = self.load_portfolio()

        # Step 2: Historical EL (always computed)
        rates = self.load_historical_rates()
        df = self.compute_el_historical(df, rates)

        # Step 3: Model-based EL (optional)
        if use_models:
            df = self.compute_el_model(df)

        # Step 4: Portfolio aggregation
        self.aggregate_portfolio(df)

        # Step 5: Comparison
        if use_models:
            self.compare_model_historical(df)

        # Step 6: Export
        print("\nExporting results...")
        self.export_results(df)

        # Step 7: Segment summaries
        print("\n--- EL by Term ---")
        term_summary = self.aggregate_by_segment(df, 'term')
        print(term_summary.to_string())

        print("\n--- EL by Purpose (Top 10) ---")
        purpose_summary = self.aggregate_by_segment(df, 'purpose').head(10)
        print(purpose_summary.to_string())

        self.loan_el = df

        print("\n" + "=" * 60)
        print("EXPECTED LOSS CALCULATION COMPLETE")
        print("=" * 60)

        return df, self.portfolio_summary, self.comparison


# ==================================================================
# Standalone Execution
# ==================================================================
if __name__ == "__main__":
    try:
        calculator = ExpectedLossCalculator()
        df, portfolio, comparison = calculator.run(use_models=True)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
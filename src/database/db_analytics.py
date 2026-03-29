import pandas as pd
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database.connection import get_engine
import config


class RiskAnalytics:
    """
    SQL-driven credit risk analytics engine.

    Reads SQL query files from src/database/queries/ and executes them
    against the PostgreSQL database. Returns pandas DataFrames for
    downstream reporting and visualization.
    """

    def __init__(self):
        self.engine = get_engine()
        if not self.engine:
            raise ConnectionError("Database connection failed. Check config.py and PostgreSQL.")

        # Resolve the queries directory (next to this file)
        self.queries_dir = Path(__file__).resolve().parent / 'queries'

        # Storage for results
        self.results = {}

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _read_query(self, filename: str) -> str:
        """Load a SQL file from the queries directory."""
        filepath = self.queries_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Query file not found: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def _execute(self, sql: str) -> pd.DataFrame:
        """Execute a SQL string and return a DataFrame."""
        return pd.read_sql(sql, self.engine)

    # ------------------------------------------------------------------
    # Public API — Individual Analyses
    # ------------------------------------------------------------------

    def portfolio_summary(self) -> pd.DataFrame:
        """
        Portfolio summary grouped by grade.
        Returns total/avg loan amounts, interest rates, FICO, income, DTI.
        """
        print("📊 Running: Portfolio Summary...")
        sql = self._read_query('portfolio_summary.sql')
        df = self._execute(sql)
        self.results['portfolio_summary'] = df
        print(f"   -> {len(df)} grade groups returned.")
        return df

    def default_rate_analysis(self) -> pd.DataFrame:
        """
        Default rate analysis by grade.
        Only considers mature loans (Fully Paid, Charged Off, Default).
        """
        print("📊 Running: Default Rate Analysis...")
        sql = self._read_query('default_rate_analysis.sql')
        df = self._execute(sql)
        self.results['default_rate_analysis'] = df
        print(f"   -> {len(df)} grade groups returned.")
        return df

    def lgd_calculation(self) -> pd.DataFrame:
        """
        Loss Given Default (LGD) by grade.
        LGD = 1 - (total_pymnt / funded_amnt) for defaulted loans.
        Includes recovery rate, median/P25/P75 LGD, and total loss.
        """
        print("📊 Running: LGD Calculation...")
        sql = self._read_query('lgd_calculation.sql')
        df = self._execute(sql)
        self.results['lgd_calculation'] = df
        print(f"   -> {len(df)} grade groups returned.")
        return df

    def expected_loss_by_segment(self) -> pd.DataFrame:
        """
        Expected Loss (EL) by grade.
        EL = PD x EAD x LGD, using historical PD and LGD per grade.
        """
        print("📊 Running: Expected Loss by Segment...")
        sql = self._read_query('el_by_segment.sql')
        df = self._execute(sql)
        self.results['expected_loss_by_segment'] = df
        print(f"   -> {len(df)} grade groups returned.")
        return df

    # ------------------------------------------------------------------
    # Pipeline: Run All
    # ------------------------------------------------------------------

    def run_all(self):
        """Execute all four analyses in sequence and store results."""
        print("=" * 60)
        print("🚀 RUNNING FULL SQL ANALYTICS PIPELINE")
        print("=" * 60)

        self.portfolio_summary()
        self.default_rate_analysis()
        self.lgd_calculation()
        self.expected_loss_by_segment()

        print("\n" + "=" * 60)
        print("✅ ALL ANALYSES COMPLETE")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_results(self, output_dir=None):
        """
        Save all results to CSV files for downstream use (Power BI, reports).

        Args:
            output_dir: Directory to save CSVs. Defaults to data/powerbi/.
        """
        if not self.results:
            print("⚠️ No results to export. Run run_all() first.")
            return

        if output_dir is None:
            output_dir = config.BASE_DIR / 'data' / 'powerbi'
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        for name, df in self.results.items():
            filepath = output_dir / f'{name}.csv'
            df.to_csv(filepath, index=False)
            print(f"   💾 Saved: {filepath} ({len(df)} rows)")

        print(f"   -> All results exported to: {output_dir}")

    def export_combined_risk_metrics(self, output_path=None):
        """
        Export a single combined risk_metrics.csv for Power BI.

        Args:
            output_path: File path for the combined CSV.
                        Defaults to data/powerbi/risk_metrics.csv.
        """
        if not self.results:
            print("⚠️ No results to export. Run run_all() first.")
            return

        if output_path is None:
            output_path = config.BASE_DIR / 'data' / 'powerbi' / 'risk_metrics.csv'
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Merge PD + LGD + EL into a single wide table by grade
        el_df = self.results.get('expected_loss_by_segment')
        lgd_df = self.results.get('lgd_calculation')

        if el_df is not None and lgd_df is not None:
            combined = el_df.merge(
                lgd_df[['grade', 'avg_recovery_rate', 'median_lgd', 'p25_lgd', 'p75_lgd']],
                on='grade',
                how='left'
            )
            combined.to_csv(output_path, index=False)
            print(f"   💾 Combined risk_metrics saved to: {output_path} ({len(combined)} rows)")
        else:
            print("⚠️ Missing EL or LGD results for combined export.")


# ------------------------------------------------------------------
# Standalone Execution!
# ------------------------------------------------------------------
if __name__ == "__main__":
    try:
        analytics = RiskAnalytics()
        analytics.run_all()
        analytics.export_results()
        analytics.export_combined_risk_metrics()
    except Exception as e:
        print(f"❌ Analytics pipeline failed: {e}")
        import traceback
        traceback.print_exc()
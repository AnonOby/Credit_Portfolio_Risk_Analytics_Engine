"""
Power BI Data Export

Exports pre-computed analytics tables from PostgreSQL to CSV files
that Power BI can directly consume.  All exports go to data/powerbi/.

This module can be run as a standalone script or imported as a utility.

Usage:
    python -m src.visualization.powerbi_export
"""

import sys
import os
from pathlib import Path

import pandas as pd

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.database import get_engine
from src.visualization.data_fetcher import DataFetcher


def _export_dir() -> Path:
    """Create and return the Power BI export directory."""
    d = _PROJECT_ROOT / "data" / "powerbi"
    d.mkdir(parents=True, exist_ok=True)
    return d


def export_all(output_dir: Path = None):
    """
    Export all analytics tables to CSV for Power BI consumption.

    Parameters
    ----------
    output_dir : Path, optional
        Target directory. Defaults to data/powerbi/.
    """
    if output_dir is None:
        output_dir = _export_dir()

    exports = {
        "portfolio_summary.csv": DataFetcher.portfolio_summary,
        "grade_distribution.csv": DataFetcher.grade_distribution,
        "loan_status_distribution.csv": DataFetcher.loan_status_distribution,
        "term_distribution.csv": DataFetcher.term_distribution,
        "purpose_distribution.csv": DataFetcher.purpose_distribution,
        "home_ownership_distribution.csv": DataFetcher.home_ownership_distribution,
        "state_distribution.csv": DataFetcher.state_distribution,
        "issuance_trend.csv": DataFetcher.issuance_trend,
        "default_rates_by_grade.csv": DataFetcher.default_rates_by_grade,
        "default_rates_over_time.csv": DataFetcher.default_rates_over_time,
        "default_by_purpose.csv": DataFetcher.default_by_purpose,
        "default_by_home_ownership.csv": DataFetcher.default_by_home_ownership,
        "default_by_dti_bucket.csv": DataFetcher.default_by_dti_bucket,
        "default_by_fico_bucket.csv": DataFetcher.default_by_fico_bucket,
        "lgd_by_grade.csv": DataFetcher.lgd_by_grade,
        "el_by_grade.csv": DataFetcher.el_by_grade,
        "concentration_metrics.csv": DataFetcher.concentration_metrics,
        "int_rate_by_grade.csv": DataFetcher.int_rate_by_grade,
        "emp_length_distribution.csv": DataFetcher.emp_length_distribution,
        "income_distribution.csv": DataFetcher.income_distribution,
    }

    print("=" * 60)
    print("Power BI Data Export")
    print("=" * 60)
    print("Target directory: {}".format(output_dir))
    print("")

    for filename, fetch_func in exports.items():
        try:
            df = fetch_func()
            filepath = output_dir / filename
            df.to_csv(filepath, index=False)
            print("  [OK] {:40s} -> {:>8,} rows".format(filename, len(df)))
        except Exception as exc:
            print("  [FAIL] {:40s} -> {}".format(filename, exc))

    print("")
    print("Done. {} files exported to {}".format(len(exports), output_dir))


# Allow running as standalone: python -m src.visualization.powerbi_export
if __name__ == "__main__":
    export_all()
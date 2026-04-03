"""
main.py - Pipeline Orchestrator
================================

End-to-end execution of the Credit Portfolio Risk Analytics Engine.

Steps:
    1. ETL Pipeline        - Extract, clean, enrich, load to PostgreSQL
    2. SQL Analytics       - Portfolio summary, default rates, LGD, EL
    3. PD Model Training   - HistGradientBoostingClassifier (AUC-ROC target)
    4. LGD Model Training  - GradientBoostingRegressor with Huber loss
    5. Vasicek ASRF Model  - Monte Carlo loss distribution, VaR @ 99.9%
    6. Power BI Export     - CSV tables for Power BI dashboard
    7. PDF Report          - LaTeX risk report generation

Usage:
    python main.py                  # Run full pipeline (steps 1-7)
    python main.py --step 3         # Run only step 3 (PD model)
    python main.py --step 3 4 5     # Run steps 3, 4, 5 only
    python main.py --skip-etl       # Skip step 1 (data already loaded)
    python main.py --list           # List all available steps

Notes:
    - Step 1 (ETL) is time-consuming (~15 min for 2.26M records).
      Use --skip-etl if the database is already populated.
    - Step 5 (Vasicek) takes ~2 min for 100K Monte Carlo scenarios.
    - Step 7 (PDF) requires a LaTeX distribution (Texmaker / pdflatex).
    - All model artifacts are saved to output/models/ and output/reports/.
"""

import sys
import os
import time
import argparse
from datetime import datetime
from pathlib import Path

# Ensure project root is on Python path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ==================================================================
# Pipeline Step Definitions
# ==================================================================

STEPS = [
    {
        "id": 1,
        "name": "ETL Pipeline",
        "description": "Extract raw CSV, clean, merge Census data, load to PostgreSQL",
    },
    {
        "id": 2,
        "name": "SQL Analytics",
        "description": "Portfolio summary, default rates, LGD, EL by segment",
    },
    {
        "id": 3,
        "name": "PD Model Training",
        "description": "HistGradientBoostingClassifier with permutation importance",
    },
    {
        "id": 4,
        "name": "LGD Model Training",
        "description": "GradientBoostingRegressor (Huber loss) with dashboard export",
    },
    {
        "id": 5,
        "name": "Vasicek ASRF Model",
        "description": "Monte Carlo simulation, VaR @ 99.9%, economic capital",
    },
    {
        "id": 6,
        "name": "Power BI Export",
        "description": "Export CSV tables to data/powerbi/ for dashboard ingestion",
    },
    {
        "id": 7,
        "name": "PDF Report",
        "description": "Generate LaTeX report with tables, metrics, and equations",
    },
]


# ==================================================================
# Step Execution Functions
# ==================================================================

def run_step_1_etl():
    """
    Step 1: ETL Pipeline.

    Orchestrates the full data extraction, cleaning, Census enrichment,
    and PostgreSQL loading pipeline via PortfolioDataLoader.
    """
    from src.etl.loader import PortfolioDataLoader

    print("\n" + "=" * 70)
    print("STEP 1/7: ETL PIPELINE")
    print("  Extract -> Clean -> Enrich (Census) -> Load to PostgreSQL")
    print("=" * 70)

    loader = PortfolioDataLoader()
    loader.run()

    print("\n   ETL pipeline completed.")


def run_step_2_analytics():
    """
    Step 2: SQL Analytics.

    Runs the four core SQL analytics queries against PostgreSQL:
    portfolio summary, default rates, LGD calculation, EL by segment.
    """
    from src.database.db_analytics import RiskAnalytics

    print("\n" + "=" * 70)
    print("STEP 2/7: SQL ANALYTICS")
    print("  Portfolio Summary | Default Rates | LGD | Expected Loss")
    print("=" * 70)

    analytics = RiskAnalytics()
    analytics.run_all()
    analytics.export_results()
    analytics.export_combined_risk_metrics()

    print("\n   SQL analytics completed. Results saved to data/powerbi/.")


def run_step_3_pd_model():
    """
    Step 3: PD Model Training.

    Trains a HistGradientBoostingClassifier on mature loans to predict
    the probability of default. Uses permutation importance for feature
    ranking (Python 3.14 sklearn compatibility). Exports dashboard JSON
    and feature importance CSV to output/models/.
    """
    from src.analytics.pd_model import PDModel

    print("\n" + "=" * 70)
    print("STEP 3/7: PD MODEL TRAINING")
    print("  HistGradientBoostingClassifier -> AUC-ROC, Feature Importance")
    print("=" * 70)

    model = PDModel()
    metrics = model.train()
    model.summary_by_grade()
    model.summary_by_segment(segment_col="term")

    print("\n   PD model metrics: AUC-ROC = {}".format(metrics.get("auc_roc", "N/A")))
    print("   Artifacts saved to output/models/.")


def run_step_4_lgd_model():
    """
    Step 4: LGD Model Training.

    Trains a GradientBoostingRegressor with Huber loss on defaulted loans
    to predict recovery rates, then derives LGD = 1 - recovery_rate.
    Exports dashboard JSON with predictions scatter data and feature
    importance CSV to output/models/.
    """
    from src.analytics.lgd_model import LGDModel

    print("\n" + "=" * 70)
    print("STEP 4/7: LGD MODEL TRAINING")
    print("  GradientBoostingRegressor (Huber) -> Recovery Rate -> LGD")
    print("=" * 70)

    model = LGDModel()
    metrics = model.train()
    model.summary_by_grade()

    print("\n   LGD model metrics: R2 = {}".format(metrics.get("recovery_r2", "N/A")))
    print("   Artifacts saved to output/models/.")


def run_step_5_vasicek():
    """
    Step 5: Vasicek ASRF Model.

    Runs Monte Carlo simulation (100K scenarios) with grade-grouped
    binomial approximation to compute the portfolio loss distribution.
    Outputs: EL, VaR @ 99.9%, Unexpected Loss, Economic Capital,
    and per-segment risk metrics. Exports dashboard JSON to output/models/.
    """
    from src.analytics.vasicek import VasicekModel

    print("\n" + "=" * 70)
    print("STEP 5/7: VASICEK ASRF MODEL")
    print("  Monte Carlo (100K scenarios) -> Loss Distribution -> VaR @ 99.9%")
    print("=" * 70)

    model = VasicekModel()
    summary, segments = model.run()

    if summary:
        print("\n   Expected Loss:    ${:,.0f}".format(summary.get("el", 0)))
        print("   VaR @ 99.9%:      ${:,.0f}".format(summary.get("var_99.9%", 0)))
        print("   Unexpected Loss:  ${:,.0f}".format(summary.get("unexpected_loss", 0)))
        print("   Econ. Capital:    ${:,.0f}".format(summary.get("economic_capital", 0)))

    print("   Artifacts saved to output/ and output/models/.")


def run_step_6_powerbi():
    """
    Step 6: Power BI Export.

    Attempts to use the comprehensive powerbi_export module first.
    If not available, falls back to RiskAnalytics.export_results()
    which exports the 4 core SQL query results as CSVs.
    """
    print("\n" + "=" * 70)
    print("STEP 6/7: POWER BI EXPORT")
    print("  CSV tables -> data/powerbi/")
    print("=" * 70)

    # Try the comprehensive export module first
    try:
        from src.visualization.powerbi_export import export_powerbi_data

        print("   Using comprehensive powerbi_export module...")
        export_powerbi_data()
        print("   Power BI data exported successfully.")
        return
    except ImportError:
        print("   powerbi_export module not found, using RiskAnalytics fallback...")
    except Exception as exc:
        print("   powerbi_export failed ({}), using fallback...".format(exc))

    # Fallback: use RiskAnalytics to export the 4 core queries
    try:
        from src.database.db_analytics import RiskAnalytics

        analytics = RiskAnalytics()
        analytics.run_all()
        analytics.export_results()
        analytics.export_combined_risk_metrics()
        print("   Power BI data exported via RiskAnalytics fallback.")
    except Exception as exc:
        print("   ERROR: Power BI export failed: {}".format(exc))


def run_step_7_report():
    """
    Step 7: PDF Report Generation.

    Generates a professional LaTeX (.tex) risk report covering all
    analytics results. If pdflatex is available, the .tex is compiled
    to PDF automatically. Otherwise, compile manually in Texmaker.
    """
    from src.visualization.pdf_report import generate_report

    print("\n" + "=" * 70)
    print("STEP 7/7: PDF REPORT GENERATION")
    print("  LaTeX report -> output/reports/")
    print("=" * 70)

    output_path = generate_report()
    print("   Report generated: {}".format(output_path))


# ==================================================================
# Step Dispatcher
# ==================================================================

STEP_RUNNERS = {
    1: run_step_1_etl,
    2: run_step_2_analytics,
    3: run_step_3_pd_model,
    4: run_step_4_lgd_model,
    5: run_step_5_vasicek,
    6: run_step_6_powerbi,
    7: run_step_7_report,
}


def run_steps(step_ids):
    """
    Execute a sequence of pipeline steps with timing and error handling.

    Parameters
    ----------
    step_ids : list[int]
        Ordered list of step IDs to execute.

    Returns
    -------
    dict
        Mapping of step_id -> {"status": "ok"|"error", "duration": float, "error": str|None}
    """
    results = {}
    pipeline_start = time.time()

    print("")
    print("#" * 70)
    print("#  CREDIT PORTFOLIO RISK ANALYTICS ENGINE")
    print("#  Pipeline Execution: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("#  Steps to run: {}".format(", ".join(str(s) for s in step_ids)))
    print("#" * 70)

    for step_id in step_ids:
        step_info = next((s for s in STEPS if s["id"] == step_id), None)
        if step_info is None:
            print("\n   WARNING: Unknown step ID {}. Skipping.".format(step_id))
            continue

        runner = STEP_RUNNERS.get(step_id)
        if runner is None:
            print("\n   WARNING: No runner for step {}. Skipping.".format(step_id))
            continue

        step_start = time.time()
        status = "ok"
        error_msg = None

        try:
            runner()
        except Exception as exc:
            status = "error"
            error_msg = str(exc)
            print("\n   ERROR in Step {} ({}): {}".format(step_id, step_info["name"], exc))
            import traceback
            traceback.print_exc()

        duration = time.time() - step_start
        results[step_id] = {
            "status": status,
            "duration": duration,
            "error": error_msg,
        }

        step_status_icon = "[OK]" if status == "ok" else "[FAIL]"
        print("\n   {} Step {}: {} ({:.1f}s)".format(
            step_status_icon, step_id, step_info["name"], duration
        ))

        # Continue pipeline even if a step fails
        if status == "error":
            print("   Continuing to next step...")

    # Summary
    total_duration = time.time() - pipeline_start

    print("")
    print("#" * 70)
    print("#  PIPELINE SUMMARY")
    print("#" * 70)

    for step_id, result in results.items():
        step_info = next((s for s in STEPS if s["id"] == step_id), {})
        icon = "[OK]  " if result["status"] == "ok" else "[FAIL]"
        line = "   {} Step {}: {} ({:.1f}s)".format(
            icon, step_id, step_info.get("name", "?"), result["duration"]
        )
        if result["error"]:
            line += " - {}".format(result["error"])
        print(line)

    n_ok = sum(1 for r in results.values() if r["status"] == "ok")
    n_fail = sum(1 for r in results.values() if r["status"] == "error")

    print("")
    print("   Total: {}/{} steps succeeded in {:.1f}s".format(
        n_ok, len(results), total_duration
    ))

    if n_fail > 0:
        print("   {} step(s) failed. Review errors above.".format(n_fail))
    else:
        print("   All steps completed successfully.")

    print("#" * 70)
    print("")

    return results


# ==================================================================
# CLI Interface
# ==================================================================

def list_steps():
    """Print all available pipeline steps."""
    print("")
    print("Available Pipeline Steps:")
    print("-" * 70)
    for step in STEPS:
        print("   Step {}: {}".format(step["id"], step["name"]))
        print("          {}".format(step["description"]))
    print("")
    print("Usage:")
    print("   python main.py                  # Full pipeline (steps 1-7)")
    print("   python main.py --step 3         # Run step 3 only")
    print("   python main.py --step 3 4 5     # Run steps 3, 4, 5")
    print("   python main.py --skip-etl       # Skip ETL, run steps 2-7")
    print("")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Credit Portfolio Risk Analytics Engine - Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  Run full pipeline (steps 1-7)
  python main.py --step 3         Run PD model training only
  python main.py --step 3 4 5     Run models and Vasicek only
  python main.py --skip-etl       Skip ETL, run steps 2-7
  python main.py --list           List available steps
        """,
    )

    parser.add_argument(
        "--step",
        type=int,
        nargs="+",
        default=None,
        help="Run specific step(s) by ID. e.g. --step 3 or --step 3 4 5",
    )
    parser.add_argument(
        "--skip-etl",
        action="store_true",
        default=False,
        help="Skip step 1 (ETL). Useful when database is already populated.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="List all available pipeline steps and exit.",
    )

    return parser.parse_args()


def main():
    """Main entry point for the pipeline orchestrator."""
    args = parse_args()

    if args.list:
        list_steps()
        return

    # Determine which steps to run
    if args.step is not None:
        # User specified explicit steps
        step_ids = sorted(set(args.step))
    elif args.skip_etl:
        # Skip ETL, run steps 2-7
        step_ids = [2, 3, 4, 5, 6, 7]
    else:
        # Full pipeline: steps 1-7
        step_ids = [1, 2, 3, 4, 5, 6, 7]

    # Execute
    results = run_steps(step_ids)

    # Exit code: 0 if all succeeded, 1 if any failed
    n_fail = sum(1 for r in results.values() if r["status"] == "error")
    sys.exit(1 if n_fail > 0 else 0)


# ==================================================================
# Standalone Execution
# ==================================================================

if __name__ == "__main__":
    main()
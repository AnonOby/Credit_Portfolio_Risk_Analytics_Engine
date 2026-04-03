"""
chart_generator.py - Static Matplotlib Chart Generator (18 Charts)
====================================================================

Generates 18 professional PNG charts from the PostgreSQL database
(loans_master table) for the Credit Portfolio Risk Analytics Engine.
All charts are saved to output/figures/.

Charts (18 total):
    --- Portfolio Overview ---
    01. portfolio_composition       - Loan count + amount by grade (dual axis)
    02. loan_status_distribution    - Loan status horizontal bar
    03. portfolio_by_term           - Portfolio by term (36 vs 60)
    04. portfolio_by_purpose        - Top 10 purposes by count

    --- Default Rate Analysis ---
    05. default_rate_by_grade       - Default rate by grade with labels
    06. default_rate_by_term        - Default rate by term
    07. default_rate_by_purpose     - Top 10 purposes by default rate
    08. default_rate_by_year        - Default rate trend by issue year

    --- LGD & Recovery ---
    09. lgd_by_grade                - LGD with IQR error bars by grade
    10. recovery_rate_by_grade      - Recovery rate by grade

    --- Expected Loss ---
    11. expected_loss_by_grade      - EL breakdown by grade
    12. risk_metrics_combined       - PD + LGD + EL combined by grade

    --- Distribution Analysis ---
    13. interest_rate_by_grade      - Interest rate percentiles by grade
    14. fico_by_grade               - FICO score percentiles by grade
    15. dti_by_grade                - DTI percentiles by grade
    16. income_by_grade             - Income percentiles by grade

    --- Time Series ---
    17. monthly_issuance            - Monthly loan volume trend
    18. vintage_default_rate        - Default rate by vintage month

Usage:
    from src.visualization.chart_generator import generate_all_charts
    generate_all_charts()

    # Or standalone:
    python -m src.visualization.chart_generator

Notes:
    - Uses .format() for string formatting (PyCharm compatibility).
    - English comments only.
    - Queries database directly; no dependency on Power BI CSV files.
    - All PNG files saved at 200 DPI.
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===================================================================
# Global Style Configuration
# ===================================================================

# Grade color palette (green -> red for A -> G)
GRADE_COLORS = {
    "A": "#1a9641",
    "B": "#73c476",
    "C": "#fed976",
    "D": "#fd8d3c",
    "E": "#e34a33",
    "F": "#b30000",
    "G": "#800026",
}

# Ordered grade list for consistent plotting
GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]

# Default figure DPI
DPI = 200

# Chart styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Mature and default status strings for SQL (same as powerbi_export.py)
_MATURE_STATUSES = (
    "'Fully Paid', 'Charged Off', 'Default', "
    "'Does not meet the credit policy. Status:Fully Paid', "
    "'Does not meet the credit policy. Status:Charged Off'"
)

_DEFAULT_STATUSES = (
    "'Charged Off', 'Default', "
    "'Does not meet the credit policy. Status:Charged Off'"
)


# ===================================================================
# Helper Functions
# ===================================================================

def _get_engine():
    """Get SQLAlchemy engine from the project connection module."""
    from src.database.connection import get_engine
    engine = get_engine()
    if engine is None:
        raise ConnectionError("Database engine returned None")
    return engine


def _get_output_dir():
    """Resolve the output directory for chart PNG files."""
    try:
        import config
        output_dir = config.BASE_DIR / "output" / "figures"
    except ImportError:
        output_dir = _PROJECT_ROOT / "output" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _query_to_df(engine, sql):
    """Execute SQL and return DataFrame. Returns empty DF on error."""
    try:
        return pd.read_sql(sql, engine)
    except Exception as exc:
        print("   WARNING: Query failed: {}".format(exc))
        return pd.DataFrame()


def _save_fig(fig, filepath, title_text=""):
    """Save figure to PNG and close."""
    fig.savefig(str(filepath), dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    size_kb = filepath.stat().st_size / 1024
    print("   [OK]  {:45s}  {:>6.0f} KB".format(filepath.name, size_kb))


def _grade_color_list(grades):
    """Return a list of colors matching the grade order."""
    return [GRADE_COLORS.get(g, "#999999") for g in grades]


def _add_source_annotation(fig, text="Source: Lending Club Loans | Credit Portfolio Risk Analytics"):
    """Add a small source annotation at bottom-right."""
    fig.text(0.99, 0.01, text, fontsize=7, color="#999999",
             ha="right", va="bottom")


def _format_millions(x, pos):
    """Format axis tick as millions: 1.5M, 2.0B, etc."""
    if abs(x) >= 1e9:
        return "${:.1f}B".format(x / 1e9)
    elif abs(x) >= 1e6:
        return "${:.1f}M".format(x / 1e6)
    elif abs(x) >= 1e3:
        return "${:.0f}K".format(x / 1e3)
    else:
        return "${:.0f}".format(x)


# ===================================================================
# Chart Functions: Portfolio Overview (4 charts)
# ===================================================================

def chart_01_portfolio_composition(engine, output_dir):
    """Portfolio composition by grade - dual axis bar (count + amount)."""
    sql = """
    SELECT
        grade,
        COUNT(*)                                      AS loan_count,
        ROUND(SUM(loan_amnt)::numeric, 2)             AS total_amount,
        ROUND(AVG(loan_amnt)::numeric, 2)             AS avg_amount
    FROM loans_master
    GROUP BY grade
    ORDER BY grade;
    """
    df = _query_to_df(engine, sql)
    if df.empty:
        return

    grades = df["grade"].tolist()
    colors = _grade_color_list(grades)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart for loan count
    bars = ax1.bar(grades, df["loan_count"], color=colors, width=0.6,
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax1.set_xlabel("Loan Grade")
    ax1.set_ylabel("Number of Loans", color="#333333")
    ax1.tick_params(axis="y", labelcolor="#333333")

    # Add count labels on bars
    for bar, count in zip(bars, df["loan_count"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 "{:,}".format(count), ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color="#333333")

    # Second axis for total amount
    ax2 = ax1.twinx()
    ax2.plot(grades, df["total_amount"], "o-", color="#1f77b4",
             linewidth=2.5, markersize=8, zorder=5)
    ax2.set_ylabel("Total Loan Amount", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_format_millions))

    ax1.set_title("Portfolio Composition by Loan Grade",
                  fontsize=15, fontweight="bold", pad=15)
    _add_source_annotation(fig)
    _save_fig(fig, output_dir / "01_portfolio_composition.png")


def chart_02_loan_status_distribution(engine, output_dir):
    """Loan status distribution - horizontal bar chart."""
    sql = """
    SELECT
        loan_status,
        COUNT(*)                                      AS loan_count,
        ROUND(SUM(loan_amnt)::numeric, 2)             AS total_amount
    FROM loans_master
    GROUP BY loan_status
    ORDER BY loan_count DESC;
    """
    df = _query_to_df(engine, sql)
    if df.empty:
        return

    # Reverse for horizontal bar (top at top)
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.5)))

    # Color by status type
    status_colors = []
    for status in df["loan_status"]:
        s_lower = status.lower()
        if "charged off" in s_lower or "default" in s_lower:
            status_colors.append("#e34a33")
        elif "fully paid" in s_lower:
            status_colors.append("#1a9641")
        elif "current" in s_lower:
            status_colors.append("#4292c6")
        else:
            status_colors.append("#999999")

    bars = ax.barh(df["loan_status"], df["loan_count"],
                   color=status_colors, edgecolor="white", linewidth=0.5,
                   height=0.7, zorder=3)

    # Add labels
    total = df["loan_count"].sum()
    for bar, count in zip(bars, df["loan_count"]):
        pct = count / total * 100
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                "  {:,}  ({:.1f}%)".format(count, pct),
                va="center", fontsize=9, color="#333333")

    ax.set_xlabel("Number of Loans")
    ax.set_title("Loan Status Distribution",
                  fontsize=15, fontweight="bold", pad=15)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, p: "{:,.0f}".format(x)))
    ax.set_xlim(0, df["loan_count"].max() * 1.25)
    _add_source_annotation(fig)
    _save_fig(fig, output_dir / "02_loan_status_distribution.png")


def chart_03_portfolio_by_term(engine, output_dir):
    """Portfolio by loan term (36 vs 60 months)."""
    sql = """
    SELECT
        term,
        COUNT(*)                                      AS loan_count,
        ROUND(SUM(loan_amnt)::numeric, 2)             AS total_amount,
        ROUND(AVG(loan_amnt)::numeric, 2)             AS avg_amount,
        ROUND(AVG(int_rate)::numeric, 2)              AS avg_int_rate,
        ROUND(AVG((fico_range_low + fico_range_high) / 2.0)::numeric, 1) AS avg_fico
    FROM loans_master
    GROUP BY term
    ORDER BY term;
    """
    df = _query_to_df(engine, sql)
    if df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ["#4292c6", "#ef6548"]
    terms = df["term"].astype(str).tolist()

    # Chart 1: Loan count
    axes[0].bar(terms, df["loan_count"], color=colors, width=0.5,
                edgecolor="white", zorder=3)
    axes[0].set_title("Number of Loans", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Count")
    for i, (_, row) in enumerate(df.iterrows()):
        axes[0].text(i, row["loan_count"], "{:,}".format(row["loan_count"]),
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, p: "{:,.0f}".format(x)))

    # Chart 2: Total amount
    axes[1].bar(terms, df["total_amount"], color=colors, width=0.5,
                edgecolor="white", zorder=3)
    axes[1].set_title("Total Loan Amount", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Amount ($)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(_format_millions))

    # Chart 3: Average interest rate + FICO (dual axis)
    ax2 = axes[2].twinx()
    bars = axes[2].bar(terms, df["avg_int_rate"], color=colors, width=0.5,
                       edgecolor="white", zorder=3, alpha=0.8, label="Avg Int Rate")
    ax2.plot(terms, df["avg_fico"], "s-", color="#333333",
             linewidth=2, markersize=10, zorder=5, label="Avg FICO")
    axes[2].set_title("Avg Interest Rate & FICO", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Interest Rate (%)")
    ax2.set_ylabel("FICO Score")
    for i, (_, row) in enumerate(df.iterrows()):
        axes[2].text(i, row["avg_int_rate"], "{:.1f}%".format(row["avg_int_rate"]),
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    fig.suptitle("Portfolio Comparison by Loan Term",
                 fontsize=15, fontweight="bold", y=1.02)
    _add_source_annotation(fig)
    fig.tight_layout()
    _save_fig(fig, output_dir / "03_portfolio_by_term.png")


def chart_04_portfolio_by_purpose(engine, output_dir):
    """Portfolio by loan purpose - top 10 horizontal bar."""
    sql = """
    SELECT
        purpose,
        COUNT(*)                                      AS loan_count,
        ROUND(SUM(loan_amnt)::numeric, 2)             AS total_amount,
        ROUND(AVG(loan_amnt)::numeric, 2)             AS avg_amount
    FROM loans_master
    WHERE purpose IS NOT NULL
    GROUP BY purpose
    ORDER BY loan_count DESC
    LIMIT 10;
    """
    df = _query_to_df(engine, sql)
    if df.empty:
        return

    df = df.iloc[::-1]  # Reverse for top item at top

    fig, ax = plt.subplots(figsize=(12, 7))

    # Blue gradient
    n = len(df)
    cmap = plt.cm.Blues
    colors = [cmap(0.35 + 0.55 * i / max(n - 1, 1)) for i in range(n)]

    bars = ax.barh(df["purpose"], df["loan_count"],
                   color=colors, edgecolor="white", linewidth=0.5,
                   height=0.7, zorder=3)

    total = df["loan_count"].sum()
    for bar, count in zip(bars, df["loan_count"]):
        pct = count / total * 100
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                "  {:,}  ({:.1f}%)".format(count, pct),
                va="center", fontsize=9, color="#333333")

    ax.set_xlabel("Number of Loans")
    ax.set_title("Top 10 Loan Purposes by Volume",
                  fontsize=15, fontweight="bold", pad=15)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, p: "{:,.0f}".format(x)))
    ax.set_xlim(0, df["loan_count"].max() * 1.2)
    _add_source_annotation(fig)
    _save_fig(fig, output_dir / "04_portfolio_by_purpose.png")


# ===================================================================
# Chart Functions: Default Rate Analysis (4 charts)
# ===================================================================

def chart_05_default_rate_by_grade(engine, output_dir):
    """Default rate by grade - bar chart with labels."""
    sql = """
    WITH mature AS (
        SELECT * FROM loans_master
        WHERE loan_status IN ({mature})
    ),
    labeled AS (
        SELECT *,
            CASE WHEN loan_status IN ({default}) THEN 1 ELSE 0 END AS is_default
        FROM mature
    )
    SELECT
        grade,
        COUNT(*)                                            AS total_mature,
        SUM(is_default)                                     AS defaulted,
        ROUND(100.0 * SUM(is_default) / COUNT(*)::numeric, 2) AS default_rate
    FROM labeled
    GROUP BY grade
    ORDER BY grade;
    """.format(mature=_MATURE_STATUSES, default=_DEFAULT_STATUSES)

    df = _query_to_df(engine, sql)
    if df.empty:
        return

    grades = df["grade"].tolist()
    colors = _grade_color_list(grades)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(grades, df["default_rate"], color=colors, width=0.6,
                  edgecolor="white", linewidth=0.5, zorder=3)

    # Labels on bars
    for bar, rate, total in zip(bars, df["default_rate"], df["total_mature"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                "{:.1f}%\n(n={:,})".format(rate, total),
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="#333333")

    ax.set_xlabel("Loan Grade")
    ax.set_ylabel("Default Rate (%)")
    ax.set_title("Default Rate by Grade (Mature Loans Only)",
                  fontsize=15, fontweight="bold", pad=15)
    ax.set_ylim(0, df["default_rate"].max() * 1.25)
    _add_source_annotation(fig)
    _save_fig(fig, output_dir / "05_default_rate_by_grade.png")


def chart_06_default_rate_by_term(engine, output_dir):
    """Default rate by term."""
    sql = """
    WITH mature AS (
        SELECT * FROM loans_master
        WHERE loan_status IN ({mature})
    ),
    labeled AS (
        SELECT *,
            CASE WHEN loan_status IN ({default}) THEN 1 ELSE 0 END AS is_default
        FROM mature
    )
    SELECT
        term,
        COUNT(*)                                            AS total_mature,
        SUM(is_default)                                     AS defaulted,
        ROUND(100.0 * SUM(is_default) / COUNT(*)::numeric, 2) AS default_rate
    FROM labeled
    GROUP BY term
    ORDER BY term;
    """.format(mature=_MATURE_STATUSES, default=_DEFAULT_STATUSES)

    df = _query_to_df(engine, sql)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#4292c6", "#ef6548"]
    terms = df["term"].astype(str).tolist()
    bars = ax.bar(terms, df["default_rate"], color=colors, width=0.4,
                  edgecolor="white", zorder=3)

    for bar, rate, total in zip(bars, df["default_rate"], df["total_mature"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                "{:.1f}%\n(n={:,})".format(rate, total),
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Loan Term (months)")
    ax.set_ylabel("Default Rate (%)")
    ax.set_title("Default Rate by Loan Term",
                  fontsize=15, fontweight="bold", pad=15)
    ax.set_ylim(0, df["default_rate"].max() * 1.25)
    _add_source_annotation(fig)
    _save_fig(fig, output_dir / "06_default_rate_by_term.png")

def chart_07_default_rate_by_purpose(engine, output_dir):
    """Default rate by purpose - top 10 horizontal bar."""
    sql = """
    WITH mature AS (
        SELECT * FROM loans_master
        WHERE loan_status IN ({mature})
    ),
    labeled AS (
        SELECT *,
            CASE WHEN loan_status IN ({default}) THEN 1 ELSE 0 END AS is_default
        FROM mature
    )
    SELECT
        purpose,
        COUNT(*)                                            AS total_mature,
        SUM(is_default)                                     AS defaulted,
        ROUND(100.0 * SUM(is_default) / COUNT(*)::numeric, 2) AS default_rate
    FROM labeled
    WHERE purpose IS NOT NULL
    GROUP BY purpose
    HAVING COUNT(*) >= 100
    ORDER BY default_rate DESC
    LIMIT 12;
    """.format(mature=_MATURE_STATUSES, default=_DEFAULT_STATUSES)

    df = _query_to_df(engine, sql)
    if df.empty:
        return

    df = df.iloc[::-1]  # Reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(12, 7))

    # Color by default rate intensity
    norm = plt.Normalize(df["default_rate"].min(), df["default_rate"].max())
    cmap = plt.cm.RdYlGn_r  # Red = high default rate
    colors = [cmap(norm(rate)) for rate in df["default_rate"]]

    bars = ax.barh(df["purpose"], df["default_rate"],
                   color=colors, edgecolor="white", linewidth=0.5,
                   height=0.7, zorder=3)

    for bar, rate, total in zip(bars, df["default_rate"], df["total_mature"]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                "  {:.1f}%  (n={:,})".format(rate, total),
                va="center", fontsize=9, color="#333333")

    ax.set_xlabel("Default Rate (%)")
    ax.set_title("Default Rate by Loan Purpose (Min 100 Loans)",
                  fontsize=15, fontweight="bold", pad=15)
    ax.set_xlim(0, df["default_rate"].max() * 1.2)
    _add_source_annotation(fig)
    _save_fig(fig, output_dir / "07_default_rate_by_purpose.png")


def chart_08_default_rate_by_year(engine, output_dir):
    """Default rate trend by issue year - line chart."""
    sql = """
    WITH mature AS (
        SELECT *,
            CASE WHEN loan_status IN ({default}) THEN 1 ELSE 0 END AS is_default
        FROM loans_master
        WHERE loan_status IN ({mature})
          AND issue_d IS NOT NULL
    )
    SELECT
        EXTRACT(YEAR FROM issue_d)::int          AS issue_year,
        COUNT(*)                                 AS total_mature,
        SUM(is_default)                          AS defaulted,
        ROUND(100.0 * SUM(is_default) / COUNT(*)::numeric, 2) AS default_rate,
        ROUND(AVG(int_rate)::numeric, 2)         AS avg_int_rate
    FROM mature
    GROUP BY EXTRACT(YEAR FROM issue_d)
    ORDER BY issue_year;
    """.format(mature=_MATURE_STATUSES, default=_DEFAULT_STATUSES)

    df = _query_to_df(engine, sql)
    if df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Line for default rate
    ax1.plot(df["issue_year"], df["default_rate"], "o-",
             color="#e34a33", linewidth=2.5, markersize=7, zorder=5,
             label="Default Rate (%)")
    ax1.fill_between(df["issue_year"], df["default_rate"],
                     alpha=0.1, color="#e34a33")
    ax1.set_xlabel("Issue Year")
    ax1.set_ylabel("Default Rate (%)", color="#e34a33")
    ax1.tick_params(axis="y", labelcolor="#e34a33")

    # Add data labels
    for _, row in df.iterrows():
        ax1.annotate("{:.1f}%".format(row["default_rate"]),
                     (row["issue_year"], row["default_rate"]),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=8, color="#e34a33")

    # Second axis for loan count
    ax2 = ax1.twinx()
    ax2.bar(df["issue_year"], df["total_mature"], width=0.6,
            color="#4292c6", alpha=0.3, zorder=2, label="Loan Count")
    ax2.set_ylabel("Number of Mature Loans", color="#4292c6")
    ax2.tick_params(axis="y", labelcolor="#4292c6")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, p: "{:,.0f}".format(x)))

    ax1.set_title("Default Rate Trend by Issue Year",
                  fontsize=15, fontweight="bold", pad=15)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    _add_source_annotation(fig)
    _save_fig(fig, output_dir / "08_default_rate_by_year.png")


# ===================================================================
# Chart Functions: LGD & Recovery (2 charts)
# ===================================================================

def chart_09_lgd_by_grade(engine, output_dir):
    """LGD by grade with IQR error bars."""
    sql = """
    SELECT
        grade,
        COUNT(*)                                                        AS total_defaulted,
        ROUND(AVG(1 - (total_pymnt / NULLIF(funded_amnt, 0)))::numeric, 4) AS avg_lgd,
        ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (
            ORDER BY 1 - total_pymnt / NULLIF(funded_amnt, 0)
        )::numeric, 4)                                                     AS p25_lgd,
        ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (
            ORDER BY 1 - total_pymnt / NULLIF(funded_amnt, 0)
        )::numeric, 4)                                                     AS median_lgd,
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (
            ORDER BY 1 - total_pymnt / NULLIF(funded_amnt, 0)
        )::numeric, 4)                                                     AS p75_lgd
    FROM loans_master
    WHERE loan_status IN ({default})
      AND funded_amnt > 0
    GROUP BY grade
    ORDER BY grade;
    """.format(default=_DEFAULT_STATUSES)

    df = _query_to_df(engine, sql)
    if df.empty:
        return

    grades = df["grade"].tolist()
    colors = _grade_color_list(grades)
    avg_lgd = df["avg_lgd"] * 100  # Convert to percentage
    p25 = df["p25_lgd"] * 100
    p75 = df["p75_lgd"] * 100
    err_lo = avg_lgd - p25
    err_hi = p75 - avg_lgd

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(grades, avg_lgd, color=colors, width=0.6,
                  edgecolor="white", linewidth=0.5, zorder=3,
                  yerr=[err_lo, err_hi], capsize=6,
                  error_kw={"ecolor": "#555555", "capthick": 1.5, "elinewidth": 1.5})

    # Labels
    for bar, avg, median in zip(bars, avg_lgd, df["median_lgd"] * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                "Mean: {:.1f}%\nMedian: {:.1f}%".format(avg, median),
                ha="center", va="bottom", fontsize=8, color="#333333")

    ax.set_xlabel("Loan Grade")
    ax.set_ylabel("Loss Given Default (%)")
    ax.set_title("LGD by Grade (with IQR)",
                  fontsize=15, fontweight="bold", pad=15)
    ax.set_ylim(0, max(p75) * 1.3)
    _add_source_annotation(fig)
    _save_fig(fig, output_dir / "09_lgd_by_grade.png")


def chart_10_recovery_rate_by_grade(engine, output_dir):
    """Recovery rate by grade."""
    sql = """
    SELECT
        grade,
        COUNT(*)                                                        AS total_defaulted,
        ROUND(AVG(total_pymnt / NULLIF(funded_amnt, 0))::numeric, 4)       AS avg_recovery,
        ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (
            ORDER BY total_pymnt / NULLIF(funded_amnt, 0)
        )::numeric, 4)                                                      AS p25_recovery,
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (
            ORDER BY total_pymnt / NULLIF(funded_amnt, 0)
        )::numeric, 4)                                                      AS p75_recovery
    FROM loans_master
    WHERE loan_status IN ({default})
      AND funded_amnt > 0
    GROUP BY grade
    ORDER BY grade;
    """.format(default=_DEFAULT_STATUSES)

    df = _query_to_df(engine, sql)
    if df.empty:
        return

    grades = df["grade"].tolist()
    # Invert grade colors (green = high recovery, red = low recovery)
    colors = _grade_color_list(grades)
    recovery = df["avg_recovery"] * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(grades, recovery, color=colors, width=0.6,
                  edgecolor="white", linewidth=0.5, zorder=3)

    for bar, rate, count in zip(bars, recovery, df["total_defaulted"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                "{:.1f}%\n(n={:,})".format(rate, count),
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Loan Grade")
    ax.set_ylabel("Recovery Rate (%)")
    ax.set_title("Recovery Rate by Grade (Defaulted Loans)",
                  fontsize=15, fontweight="bold", pad=15)
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color="#999999", linestyle=":", linewidth=1, alpha=0.7)
    _add_source_annotation(fig)
    _save_fig(fig, output_dir / "10_recovery_rate_by_grade.png")


# ===================================================================
# Chart Functions: Expected Loss (2 charts)
# ===================================================================

def chart_11_expected_loss_by_grade(engine, output_dir):
    """Expected loss breakdown by grade - stacked bar."""
    sql = """
    WITH mature AS (
        SELECT * FROM loans_master
        WHERE loan_status IN ({mature})
    ),
    pd_tbl AS (
        SELECT grade,
            ROUND(SUM(CASE WHEN loan_status IN ({default}) THEN 1 ELSE 0 END)::numeric
                  / COUNT(*), 6) AS pd
        FROM mature GROUP BY grade
    ),
    lgd_tbl AS (
        SELECT grade,
            ROUND(AVG(1 - (total_pymnt / NULLIF(funded_amnt, 0)))::numeric, 6) AS lgd
        FROM loans_master
        WHERE loan_status IN ({default}) AND funded_amnt > 0
        GROUP BY grade
    ),
    seg AS (
        SELECT m.grade,
            COUNT(*)                                AS loan_count,
            ROUND(AVG(m.funded_amnt)::numeric, 2)   AS avg_ead,
            ROUND(SUM(m.funded_amnt)::numeric, 2)   AS total_exposure,
            p.pd, l.lgd,
            ROUND(p.pd * l.lgd, 6)                 AS el_rate
        FROM mature m
        JOIN pd_tbl p ON m.grade = p.grade
        JOIN lgd_tbl l ON m.grade = l.grade
        GROUP BY m.grade, p.pd, l.lgd
    )
    SELECT grade, loan_count,
        ROUND(pd * 100, 2)         AS pd_pct,
        ROUND(lgd * 100, 2)        AS lgd_pct,
        total_exposure,
        ROUND(el_rate * total_exposure, 2) AS total_el
    FROM seg ORDER BY grade;
    """.format(mature=_MATURE_STATUSES, default=_DEFAULT_STATUSES)

    df = _query_to_df(engine, sql)
    if df.empty:
        return

    grades = df["grade"].tolist()
    x = np.arange(len(grades))
    width = 0.6

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: EL amount by grade
    colors = _grade_color_list(grades)
    bars = axes[0].bar(x, df["total_el"], width, color=colors,
                       edgecolor="white", zorder=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(grades)
    axes[0].set_xlabel("Loan Grade")
    axes[0].set_ylabel("Expected Loss ($)")
    axes[0].set_title("Total Expected Loss by Grade", fontsize=12, fontweight="bold")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(_format_millions))

    for bar, el in zip(bars, df["total_el"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     "${:,.0f}".format(el),
                     ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Right: PD and LGD comparison by grade
    w = 0.35
    axes[1].bar(x - w / 2, df["pd_pct"], w, color="#4292c6",
                edgecolor="white", zorder=3, label="PD (%)")
    axes[1].bar(x + w / 2, df["lgd_pct"], w, color="#ef6548",
                edgecolor="white", zorder=3, label="LGD (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(grades)
    axes[1].set_xlabel("Loan Grade")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_title("PD vs LGD by Grade", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper left")

    fig.suptitle("Expected Loss Analysis by Grade",
                 fontsize=15, fontweight="bold", y=1.02)
    _add_source_annotation(fig)
    fig.tight_layout()
    _save_fig(fig, output_dir / "11_expected_loss_by_grade.png")


def chart_12_risk_metrics_combined(engine, output_dir):
    """Combined PD + LGD + EL rate by grade - grouped line/bar."""
    sql = """
    WITH mature AS (
        SELECT * FROM loans_master
        WHERE loan_status IN ({mature})
    ),
    pd_tbl AS (
        SELECT grade,
            ROUND(SUM(CASE WHEN loan_status IN ({default}) THEN 1 ELSE 0 END)::numeric
                  / COUNT(*), 6) AS pd
        FROM mature GROUP BY grade
    ),
    lgd_tbl AS (
        SELECT grade,
            ROUND(AVG(1 - (total_pymnt / NULLIF(funded_amnt, 0)))::numeric, 4) AS avg_lgd
        FROM loans_master
        WHERE loan_status IN ({default}) AND funded_amnt > 0
        GROUP BY grade
    )
    SELECT
        p.grade,
        ROUND(p.pd * 100, 2)             AS pd_pct,
        ROUND(l.avg_lgd * 100, 2)        AS lgd_pct,
        ROUND(p.pd * l.avg_lgd * 100, 4) AS el_rate_pct
    FROM pd_tbl p
    JOIN lgd_tbl l ON p.grade = l.grade
    ORDER BY p.grade;
    """.format(mature=_MATURE_STATUSES, default=_DEFAULT_STATUSES)

    df = _query_to_df(engine, sql)
    if df.empty:
        return

    grades = df["grade"].tolist()
    x = np.arange(len(grades))

    fig, ax = plt.subplots(figsize=(12, 6))

    # PD bars
    ax.bar(x - 0.25, df["pd_pct"], 0.25, color="#4292c6",
           edgecolor="white", zorder=3, label="PD (%)")
    # LGD bars
    ax.bar(x, df["lgd_pct"], 0.25, color="#ef6548",
           edgecolor="white", zorder=3, label="LGD (%)")
    # EL rate line
    ax2 = ax.twinx()
    ax2.plot(x, df["el_rate_pct"], "D-", color="#1a9641",
             linewidth=2.5, markersize=8, zorder=5, label="EL Rate (%)")
    ax2.set_ylabel("Expected Loss Rate (%)", color="#1a9641")
    ax2.tick_params(axis="y", labelcolor="#1a9641")

    ax.set_xticks(x)
    ax.set_xticklabels(grades)
    ax.set_xlabel("Loan Grade")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Risk Metrics Dashboard by Grade: PD, LGD, and EL Rate",
                  fontsize=14, fontweight="bold", pad=15)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    _add_source_annotation(fig)
    _save_fig(fig, output_dir / "12_risk_metrics_combined.png")


# ===================================================================
# Chart Functions: Distribution Analysis (4 charts)
# ===================================================================

def _draw_percentile_ranges(ax, df, grade_col, val_col, colors, title, ylabel):
    """
    Draw a percentile range chart (min-p25-median-p75-max) per grade.
    This is a lightweight alternative to boxplots that uses aggregated data.
    """
    grades = df[grade_col].tolist()
    n = len(grades)
    x = np.arange(n)

    # Draw range bars (min to p75)
    for i, (_, row) in enumerate(df.iterrows()):
        # Thin line from min to max
        ax.plot([i, i], [row["p25"], row["p75"]],
                color=colors[i], linewidth=12, solid_capstyle="round",
                alpha=0.25, zorder=2)
        # Thick line for IQR (p25 to p75)
        ax.plot([i, i], [row["p25"], row["p75"]],
                color=colors[i], linewidth=6, solid_capstyle="round",
                alpha=0.5, zorder=3)
        # Median marker
        ax.plot(i, row["median"], "D", color=colors[i],
                markersize=10, zorder=5, markeredgecolor="white",
                markeredgewidth=1.5)

    # Mean line connecting grades
    ax.plot(x, df["avg"], "o--", color="#333333", linewidth=1.5,
            markersize=6, alpha=0.7, zorder=4, label="Mean")

    ax.set_xticks(x)
    ax.set_xticklabels(grades)
    ax.set_xlabel("Loan Grade")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="best")


def chart_13_interest_rate_by_grade(engine, output_dir):
    """Interest rate distribution by grade - percentile ranges."""
    sql = """
    SELECT
        grade,
        ROUND(AVG(int_rate)::numeric, 2) AS avg,
        ROUND(MIN(int_rate)::numeric, 2) AS min_val,
        ROUND(MAX(int_rate)::numeric, 2) AS max_val,
        ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY int_rate)::numeric, 2) AS p25,
        ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY int_rate)::numeric, 2) AS median,
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY int_rate)::numeric, 2) AS p75
    FROM loans_master WHERE int_rate IS NOT NULL
    GROUP BY grade ORDER BY grade;
    """
    df = _query_to_df(engine, sql)
    if df.empty:
        return

    colors = _grade_color_list(df["grade"].tolist())
    fig, ax = plt.subplots(figsize=(12, 6))
    _draw_percentile_ranges(ax, df, "grade", "avg", colors,
                            "Interest Rate Distribution by Grade",
                            "Interest Rate (%)")
    _add_source_annotation(fig)
    _save_fig(fig, output_dir / "13_interest_rate_by_grade.png")


def chart_14_fico_by_grade(engine, output_dir):
    """FICO score distribution by grade - percentile ranges."""
    sql = """
    SELECT
        grade,
        ROUND(AVG((fico_range_low + fico_range_high) / 2.0)::numeric, 1) AS avg,
        ROUND(MIN(fico_range_low)::numeric, 1) AS min_val,
        ROUND(MAX(fico_range_high)::numeric, 1) AS max_val,
        ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (
            ORDER BY (fico_range_low + fico_range_high) / 2.0
        )::numeric, 1) AS p25,
        ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (
            ORDER BY (fico_range_low + fico_range_high) / 2.0
        )::numeric, 1) AS median,
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (
            ORDER BY (fico_range_low + fico_range_high) / 2.0
        )::numeric, 1) AS p75
    FROM loans_master
    GROUP BY grade ORDER BY grade;
    """
    df = _query_to_df(engine, sql)
    if df.empty:
        return

    colors = _grade_color_list(df["grade"].tolist())
    fig, ax = plt.subplots(figsize=(12, 6))
    _draw_percentile_ranges(ax, df, "grade", "avg", colors,
                            "FICO Score Distribution by Grade",
                            "FICO Score")
    _add_source_annotation(fig)
    _save_fig(fig, output_dir / "14_fico_by_grade.png")


def chart_15_dti_by_grade(engine, output_dir):
    """DTI distribution by grade - percentile ranges."""
    sql = """
    SELECT
        grade,
        ROUND(AVG(dti)::numeric, 2) AS avg,
        ROUND(MIN(dti)::numeric, 2) AS min_val,
        ROUND(MAX(dti)::numeric, 2) AS max_val,
        ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY dti)::numeric, 2) AS p25,
        ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY dti)::numeric, 2) AS median,
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY dti)::numeric, 2) AS p75
    FROM loans_master WHERE dti IS NOT NULL
    GROUP BY grade ORDER BY grade;
    """
    df = _query_to_df(engine, sql)
    if df.empty:
        return

    colors = _grade_color_list(df["grade"].tolist())
    fig, ax = plt.subplots(figsize=(12, 6))
    _draw_percentile_ranges(ax, df, "grade", "avg", colors,
                            "Debt-to-Income (DTI) Distribution by Grade",
                            "DTI Ratio (%)")
    _add_source_annotation(fig)
    _save_fig(fig, output_dir / "15_dti_by_grade.png")


def chart_16_income_by_grade(engine, output_dir):
    """Annual income distribution by grade - percentile ranges."""
    sql = """
    SELECT
        grade,
        ROUND(AVG(annual_inc)::numeric, 2) AS avg,
        ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY annual_inc)::numeric, 2) AS p25,
        ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY annual_inc)::numeric, 2) AS median,
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY annual_inc)::numeric, 2) AS p75,
        ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY annual_inc)::numeric, 2) AS p95
    FROM loans_master
    WHERE annual_inc IS NOT NULL AND annual_inc > 0
    GROUP BY grade ORDER BY grade;
    """
    df = _query_to_df(engine, sql)
    if df.empty:
        return

    colors = _grade_color_list(df["grade"].tolist())
    fig, ax = plt.subplots(figsize=(12, 6))
    _draw_percentile_ranges(ax, df, "grade", "avg", colors,
                            "Annual Income Distribution by Grade",
                            "Annual Income ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_format_millions))
    _add_source_annotation(fig)
    _save_fig(fig, output_dir / "16_income_by_grade.png")


# ===================================================================
# Chart Functions: Time Series (2 charts)
# ===================================================================

def chart_17_monthly_issuance(engine, output_dir):
    """Monthly loan issuance trend - area chart."""
    sql = """
    SELECT
        TO_CHAR(issue_d, 'YYYY-MM')                          AS month,
        COUNT(*)                                              AS loans_issued,
        ROUND(SUM(loan_amnt)::numeric, 2)                     AS total_amount,
        ROUND(AVG(loan_amnt)::numeric, 2)                     AS avg_loan_amount
    FROM loans_master
    WHERE issue_d IS NOT NULL
    GROUP BY TO_CHAR(issue_d, 'YYYY-MM')
    ORDER BY month;
    """
    df = _query_to_df(engine, sql)
    if df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Area chart for loan count
    ax1.fill_between(range(len(df)), df["loans_issued"],
                     alpha=0.3, color="#4292c6", zorder=2)
    ax1.plot(range(len(df)), df["loans_issued"],
             color="#4292c6", linewidth=1.5, zorder=3)
    ax1.set_ylabel("Number of Loans", color="#4292c6")
    ax1.tick_params(axis="y", labelcolor="#4292c6")

    # Second axis for total amount
    ax2 = ax1.twinx()
    ax2.plot(range(len(df)), df["total_amount"],
             color="#ef6548", linewidth=1.5, zorder=3, alpha=0.8)
    ax2.set_ylabel("Total Loan Amount ($)", color="#ef6548")
    ax2.tick_params(axis="y", labelcolor="#ef6548")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_format_millions))

    # X-axis labels (show every N months to avoid crowding)
    n_labels = min(12, len(df))
    step = max(1, len(df) // n_labels)
    tick_positions = list(range(0, len(df), step))
    tick_labels = [df["month"].iloc[i] for i in tick_positions]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax1.set_title("Monthly Loan Issuance Trend",
                  fontsize=15, fontweight="bold", pad=15)
    ax1.set_xlabel("Month")

    _add_source_annotation(fig)
    fig.tight_layout()
    _save_fig(fig, output_dir / "17_monthly_issuance.png")


def chart_18_vintage_default_rate(engine, output_dir):
    """Default rate by vintage month - line chart."""
    sql = """
    WITH mature AS (
        SELECT *,
            CASE WHEN loan_status IN ({default}) THEN 1 ELSE 0 END AS is_default
        FROM loans_master
        WHERE loan_status IN ({mature})
          AND issue_d IS NOT NULL
    )
    SELECT
        TO_CHAR(issue_d, 'YYYY-MM')                          AS vintage_month,
        COUNT(*)                                              AS total_loans,
        SUM(is_default)                                       AS defaulted,
        ROUND(100.0 * SUM(is_default) / COUNT(*)::numeric, 2) AS default_rate
    FROM mature
    GROUP BY TO_CHAR(issue_d, 'YYYY-MM')
    HAVING COUNT(*) >= 50
    ORDER BY vintage_month;
    """.format(mature=_MATURE_STATUSES, default=_DEFAULT_STATUSES)

    df = _query_to_df(engine, sql)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    # Line chart for default rate
    ax.plot(range(len(df)), df["default_rate"],
            "o-", color="#e34a33", linewidth=1.5, markersize=4, zorder=5)
    ax.fill_between(range(len(df)), df["default_rate"],
                     alpha=0.15, color="#e34a33")

    # Add a rolling average (3-month window) if enough data points
    if len(df) >= 6:
        rolling = df["default_rate"].rolling(window=3, center=True, min_periods=1).mean()
        ax.plot(range(len(df)), rolling, "--",
                color="#333333", linewidth=2, alpha=0.7, zorder=4,
                label="3-Month Moving Avg")
        ax.legend(loc="upper left")

    # X-axis labels
    n_labels = min(12, len(df))
    step = max(1, len(df) // n_labels)
    tick_positions = list(range(0, len(df), step))
    tick_labels = [df["vintage_month"].iloc[i] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax.set_xlabel("Vintage Month")
    ax.set_ylabel("Default Rate (%)")
    ax.set_title("Vintage Default Rate by Issue Month (Mature Loans, Min 50)",
                  fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(bottom=0)

    _add_source_annotation(fig)
    fig.tight_layout()
    _save_fig(fig, output_dir / "18_vintage_default_rate.png")


# ===================================================================
# Main Generator
# ===================================================================

# Ordered list of all chart functions
ALL_CHARTS = [
    # Portfolio Overview
    ("01_portfolio_composition",      chart_01_portfolio_composition),
    ("02_loan_status_distribution",   chart_02_loan_status_distribution),
    ("03_portfolio_by_term",          chart_03_portfolio_by_term),
    ("04_portfolio_by_purpose",       chart_04_portfolio_by_purpose),
    # Default Rate
    ("05_default_rate_by_grade",      chart_05_default_rate_by_grade),
    ("06_default_rate_by_term",       chart_06_default_rate_by_term),
    ("07_default_rate_by_purpose",    chart_07_default_rate_by_purpose),
    ("08_default_rate_by_year",       chart_08_default_rate_by_year),
    # LGD & Recovery
    ("09_lgd_by_grade",               chart_09_lgd_by_grade),
    ("10_recovery_rate_by_grade",     chart_10_recovery_rate_by_grade),
    # Expected Loss
    ("11_expected_loss_by_grade",     chart_11_expected_loss_by_grade),
    ("12_risk_metrics_combined",      chart_12_risk_metrics_combined),
    # Distribution
    ("13_interest_rate_by_grade",     chart_13_interest_rate_by_grade),
    ("14_fico_by_grade",              chart_14_fico_by_grade),
    ("15_dti_by_grade",               chart_15_dti_by_grade),
    ("16_income_by_grade",            chart_16_income_by_grade),
    # Time Series
    ("17_monthly_issuance",           chart_17_monthly_issuance),
    ("18_vintage_default_rate",       chart_18_vintage_default_rate),
]


def generate_all_charts(engine=None, output_dir=None):
    """
    Generate all 18 charts and save to output/figures/.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine, optional
        Pre-connected engine. If None, creates one via get_engine().
    output_dir : Path or str, optional
        Output directory for PNG files. Defaults to output/figures/.

    Returns
    -------
    dict
        Summary: {chart_name: {"status": "ok"|"skip", "path": str}}
    """
    if engine is None:
        engine = _get_engine()

    if output_dir is None:
        output_dir = _get_output_dir()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print("   Generating 18 charts -> {}".format(output_dir))
    print("   " + "-" * 60)

    results = {}
    n_ok = 0
    n_skip = 0

    for idx, (name, func) in enumerate(ALL_CHARTS, 1):
        tag = "[{:02d}/18]".format(idx)
        try:
            func(engine, output_dir)
            results[name] = {"status": "ok"}
            n_ok += 1
        except Exception as exc:
            print("   {} {:40s}  FAILED: {}".format(tag, name, exc))
            results[name] = {"status": "error", "error": str(exc)}
            n_skip += 1

    print("   " + "-" * 60)
    print("   Charts generated: {}/18".format(n_ok))
    print("   Skipped/Failed:   {}".format(n_skip))
    print("   Output directory: {}".format(output_dir))
    print("")

    return results


# ===================================================================
# Standalone Execution
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  STATIC CHART GENERATOR (18 Charts)")
    print("=" * 60)

    try:
        results = generate_all_charts()
        n_ok = sum(1 for r in results.values() if r["status"] == "ok")
        print("   Done. {}/18 charts generated.".format(n_ok))
    except Exception as exc:
        print("   FATAL: {}".format(exc))
        import traceback
        traceback.print_exc()
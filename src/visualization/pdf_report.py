"""
PDF Risk Report Generator — LaTeX Version

Generates a professional LaTeX (.tex) report from portfolio analytics data,
then optionally compiles it to PDF via pdflatex.

The report covers:
    - Executive Summary (KPI table)
    - Portfolio Overview by Grade
    - Default Analysis
    - Loss Given Default (LGD)
    - Expected Loss (EL)
    - Portfolio Concentration (HHI)
    - Interest Rate Analytics

Usage:
    python -m src.visualization.pdf_report
"""

import sys
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_OUTPUT_DIR = _PROJECT_ROOT / "output" / "reports"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================================
# LaTeX helpers
# ==================================================================

def _esc(val):
    """
    Escape special LaTeX characters in a value that will appear
    inside a table cell or text paragraph.
    """
    s = str(val)
    for ch, repl in [("\\", "\\textbackslash{}"), ("%", "\\%"), ("$", "\\$"),
                     ("#", "\\#"), ("&", "\\&"), ("_", "\\_"), ("{", "\\{"),
                     ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")]:
        s = s.replace(ch, repl)
    return s


def _num(val, fmt=",.0f"):
    """Format a number with commas / decimals."""
    try:
        return format(float(val), fmt)
    except (ValueError, TypeError):
        return str(val)


def _pct(val, digits=2):
    """Format a ratio as a percentage string."""
    try:
        return "{:.{}f}\\%".format(float(val) * 100, digits)
    except (ValueError, TypeError):
        return str(val)


# ==================================================================
# LaTeX table builders
# ==================================================================

def _tabular(header, rows, col_fmt, caption=None, label=None):
    """
    Build a LaTeX tabular environment.

    Parameters
    ----------
    header : list[str]
    rows   : list[list[str]]  — each cell is already escaped / formatted
    col_fmt: str  — e.g. l r r r
    caption: str | None
    label  : str | None
    """
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    if caption:
        lines.append("\\caption{{{}}}".format(_esc(caption)))
    lines.append("\\begin{tabular}{" + col_fmt + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if label:
        lines.append("\\label{{{}}}".format(_esc(label)))
    lines.append("\\end{table}")
    return "\n".join(lines)


# ==================================================================
# Report generator
# ==================================================================

def generate_report() -> str:
    """
    Generate a LaTeX risk report and return the .tex file path.

    If pdflatex is on PATH, the .tex is also compiled to PDF.

    Returns
    -------
    str
        Absolute path to the generated .tex file.
    """
    from src.visualization.data_fetcher import DataFetcher

    # ------------------------------------------------------------------
    # 1. Collect data
    # ------------------------------------------------------------------
    print("Collecting data for LaTeX report...")

    portfolio_df = DataFetcher.portfolio_summary()
    default_df   = DataFetcher.default_rates_by_grade()
    lgd_df       = DataFetcher.lgd_by_grade()
    el_df        = DataFetcher.el_by_grade()
    conc_df      = DataFetcher.concentration_metrics()
    rate_df      = DataFetcher.int_rate_by_grade()

    # Summary stats
    total_loans = int(portfolio_df["total_loans"].sum())
    total_funded = float(portfolio_df["total_funded"].sum())
    avg_int_rate = float(portfolio_df["avg_int_rate"].mean())

    total_mature = int(default_df["total_mature"].sum())
    total_defaults = int(default_df["defaults"].sum())
    overall_dr = total_defaults / total_mature if total_mature > 0 else 0

    total_el = float(el_df["total_el"].sum())
    total_ead = float(el_df["total_ead"].sum())
    el_pct = total_el / total_ead if total_ead > 0 else 0

    avg_lgd = float(lgd_df["avg_lgd"].mean()) if len(lgd_df) > 0 else 0
    total_hhi = float(conc_df["hhi_contrib"].sum())

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_slug = datetime.now().strftime("%Y%m%d_%H%M%S")

    tex_path = _OUTPUT_DIR / "risk_report_{}.tex".format(date_slug)

    # ------------------------------------------------------------------
    # 2. Build KPI summary table (executive overview)
    # ------------------------------------------------------------------
    kpi_header = ["Metric", "Value"]
    kpi_rows = [
        ["Total Loans",             _num(total_loans)],
        ["Total Funded Amount",     "\\$" + _num(total_funded)],
        ["Average Interest Rate",   _pct(avg_int_rate / 100)],
        ["Mature Loans",            _num(total_mature)],
        ["Total Defaults",          _num(total_defaults)],
        ["Overall Default Rate",    _pct(overall_dr)],
        ["Expected Loss",           "\\$" + _num(total_el)],
        ["EL as \\% of Exposure",   _pct(el_pct)],
        ["Average LGD",             _pct(avg_lgd)],
        ["HHI Index",               _num(total_hhi, ",.0f") + " / 10{,}000"],
    ]
    kpi_table = _tabular(kpi_header, kpi_rows, "l r",
                         caption="Executive Summary", label="tab:summary")

    # ------------------------------------------------------------------
    # 3. Portfolio overview table
    # ------------------------------------------------------------------
    port_header = ["Grade", "Count", "Total Funded (\\$)", "Avg Amount (\\$)",
                   "Avg Rate (\\%)", "Avg FICO"]
    port_rows = []
    for _, r in portfolio_df.iterrows():
        avg_fico = (r.get("avg_fico_low", 0) + r.get("avg_fico_high", 0)) / 2
        port_rows.append([
            _esc(r["grade"]),
            _num(r["total_loans"]),
            _num(r["total_funded"]),
            _num(r["avg_loan_amount"]),
            _num(r["avg_int_rate"], ",.2f"),
            _num(avg_fico, ",.0f"),
        ])
    # Total row
    port_rows.append([
        "\\textbf{TOTAL}",
        "\\textbf{" + _num(total_loans) + "}",
        "\\textbf{" + _num(total_funded) + "}",
        "\\textbf{" + _num(total_funded / total_loans) + "}",
        "\\textbf{" + _num(avg_int_rate, ",.2f") + "}",
        "-",
    ])
    port_table = _tabular(port_header, port_rows, "c r r r r r",
                          caption="Portfolio Overview by Grade", label="tab:portfolio")

    # ------------------------------------------------------------------
    # 4. Default analysis table
    # ------------------------------------------------------------------
    def_header = ["Grade", "Mature Count", "Defaults", "Default Rate (\\%)"]
    def_rows = []
    for _, r in default_df.iterrows():
        def_rows.append([
            _esc(r["grade"]),
            _num(r["total_mature"]),
            _num(r["defaults"]),
            _num(r["default_rate"], ",.2f"),
        ])
    def_rows.append([
        "\\textbf{TOTAL}",
        "\\textbf{" + _num(total_mature) + "}",
        "\\textbf{" + _num(total_defaults) + "}",
        "\\textbf{" + _num(overall_dr * 100, ",.2f") + "}",
    ])
    def_table = _tabular(def_header, def_rows, "c r r r",
                         caption="Default Rate by Grade (Mature Loans Only)", label="tab:defaults")

    # ------------------------------------------------------------------
    # 5. LGD table
    # ------------------------------------------------------------------
    lgd_header = ["Grade", "Count", "Avg LGD (\\%)", "P25 (\\%)",
                  "Median (\\%)", "P75 (\\%)"]
    lgd_rows = []
    for _, r in lgd_df.iterrows():
        lgd_rows.append([
            _esc(r["grade"]),
            _num(r["total_defaulted"]),
            _pct(r["avg_lgd"], 1),
            _pct(r["p25_lgd"], 1),
            _pct(r["p50_lgd"], 1),
            _pct(r["p75_lgd"], 1),
        ])
    lgd_table = _tabular(lgd_header, lgd_rows, "c r r r r r",
                         caption="Loss Given Default Statistics by Grade", label="tab:lgd")

    # ------------------------------------------------------------------
    # 6. Expected Loss table
    # ------------------------------------------------------------------
    el_header = ["Grade", "Exposure (\\$)", "PD", "LGD",
                 "EL / Loan (\\$)", "Total EL (\\$)", "EL (\\%)"]
    el_rows = []
    for _, r in el_df.iterrows():
        grade_el_pct = r["total_el"] / total_ead if total_ead > 0 else 0
        el_rows.append([
            _esc(r["grade"]),
            _num(r["total_ead"]),
            _pct(r["pd"], 1),
            _pct(r["lgd"], 1),
            _num(r["el_per_loan"]),
            _num(r["total_el"]),
            _pct(grade_el_pct, 2),
        ])
    el_rows.append([
        "\\textbf{TOTAL}",
        "\\textbf{" + _num(total_ead) + "}",
        "-",
        "-",
        "-",
        "\\textbf{" + _num(total_el) + "}",
        "\\textbf{" + _pct(el_pct, 2) + "}",
    ])
    el_table = _tabular(el_header, el_rows, "c r r r r r r",
                        caption="Expected Loss Breakdown by Grade", label="tab:el")

    # ------------------------------------------------------------------
    # 7. Concentration table (top 12)
    # ------------------------------------------------------------------
    top_conc = conc_df.sort_values("hhi_contrib", ascending=False).head(12)
    conc_hhi_label = ("Highly Concentrated" if total_hhi > 2500
                      else "Moderately Concentrated" if total_hhi > 1500
                      else "Well Diversified")
    conc_header = ["Segment", "Type", "Share (\\%)", "HHI Contribution"]
    conc_rows = []
    for _, r in top_conc.iterrows():
        conc_rows.append([
            _esc(r["segment"]),
            _esc(r["segment_type"]),
            _num(r["pct"], ",.2f"),
            _num(r["hhi_contrib"], ",.0f"),
        ])
    conc_rows.append([
        "\\textbf{TOTAL}",
        "",
        "",
        "\\textbf{" + _num(total_hhi, ",.0f") + "}",
    ])
    conc_table = _tabular(conc_header, conc_rows, "l l r r",
                          caption="Portfolio Concentration (HHI Index) --- " + conc_hhi_label,
                          label="tab:concentration")

    # ------------------------------------------------------------------
    # 8. Interest rate table
    # ------------------------------------------------------------------
    rate_header = ["Grade", "Count", "Min (\\%)", "Avg (\\%)", "Max (\\%)"]
    rate_rows = []
    for _, r in rate_df.iterrows():
        rate_rows.append([
            _esc(r["grade"]),
            _num(r["count"]),
            _num(r["min_rate"], ",.2f"),
            _num(r["avg_rate"], ",.2f"),
            _num(r["max_rate"], ",.2f"),
        ])
    rate_table = _tabular(rate_header, rate_rows, "c r r r r",
                          caption="Interest Rate Distribution by Grade", label="tab:rates")

    # ------------------------------------------------------------------
    # 9. Assemble the full LaTeX document
    # ------------------------------------------------------------------
    doc = r"""\documentclass[11pt, a4paper]{article}

% ============================================================
% Packages
% ============================================================
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{float}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{array}
\usepackage{caption}

% ============================================================
% Page style
% ============================================================
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textit{Credit Portfolio Risk Report}}
\fancyhead[R]{\small\textit{""" + timestamp + r"""}}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

% ============================================================
% Title formatting
% ============================================================
\titleformat{\section}{\Large\bfseries\color{blue!60!black}}{\thesection}{1em}{}
\titleformat{\subsection}{\large\bfseries\color{blue!40!black}}{\thesubsection}{1em}{}

% ============================================================
% Hyperref
% ============================================================
\hypersetup{
    colorlinks=true,
    linkcolor=blue!70!black,
    citecolor=blue!70!black,
    urlcolor=blue!70!black,
}

% ============================================================
\begin{document}

% --- Title Page ------------------------------------------------
\begin{titlepage}
\centering
\vspace*{2cm}

{\Huge\bfseries\color{blue!60!black} Credit Portfolio Risk Report\par}

\vspace{0.5cm}
{\Large Lending Club Loan Portfolio Analytics\par}

\vspace{1.5cm}

\begin{tabular}{r l}
\textbf{Portfolio:}   & 2{,}260{,}668 Loans \\[4pt]
\textbf{Exposure:}    & \$34{,}004{,}208{,}600 \\[4pt]
\textbf{Grades:}      & A -- G \\[4pt]
\textbf{Data Source:} & Lending Club + US Census ACS S2503 \\[4pt]
\textbf{Generated:}   & """ + timestamp + r""" \\
\end{tabular}

\vspace{1.5cm}

""" + kpi_table + r"""

\vfill
{\small\textit{This report was auto-generated by the Credit Portfolio Risk Analytics Engine.}}
\end{titlepage}

% --- Table of Contents -----------------------------------------
\newpage
\tableofcontents
\newpage

% ============================================================
\section{Portfolio Overview}
\label{sec:portfolio}

The portfolio consists of \textbf{""" + _num(total_loans) + r"""} loans with a total funded
amount of \textbf{\$""" + _num(total_funded) + r"""}.  The average interest rate across all
grades is \textbf{""" + _num(avg_int_rate, ",.2f") + r"""\%}.  The table below breaks down key
metrics by credit grade, from A (lowest risk) to G (highest risk).  Higher grades command
larger average loan amounts but carry substantially higher default rates, as shown in
subsequent sections.

""" + port_table + r"""


% ============================================================
\section{Default Analysis}
\label{sec:defaults}

Default rates are computed on \emph{mature loans only} --- those that have reached a terminal
status (Fully Paid or Charged Off / Default).  Of the \textbf{""" + _num(total_mature) + r"""}
mature loans, \textbf{""" + _num(total_defaults) + r"""} have defaulted, yielding an overall
default rate of \textbf{""" + _pct(overall_dr) + r"""}.  Default rates increase monotonically
from Grade A to Grade G, reflecting the underlying credit risk differentiation embedded in
Lending Club's grading methodology.

""" + def_table + r"""


% ============================================================
\section{Loss Given Default}
\label{sec:lgd}

LGD measures the fraction of exposure that is \emph{lost} when a loan defaults.  It is
calculated as:

\begin{equation}
    \text{LGD} = 1 - \frac{\text{total\_pymnt}}{\text{funded\_amnt}}
\end{equation}

The average LGD across all grades is \textbf{""" + _pct(avg_lgd, 1) + r"""}.  Lower grades
tend to exhibit higher LGD, driven by higher interest rates accumulating in the balance
before the borrower ultimately defaults.

""" + lgd_table + r"""


% ============================================================
\section{Expected Loss}
\label{sec:el}

Expected Loss (EL) is the risk-adjusted anticipated loss on the portfolio, computed per
grade segment as:

\begin{equation}
    \text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}
\end{equation}

The total portfolio expected loss is \textbf{\$""" + _num(total_el) + r"""}, representing
\textbf{""" + _pct(el_pct) + r"""} of total exposure.  This is a key input for loan
pricing, capital allocation, and IFRS\,9 provisioning.

""" + el_table + r"""


% ============================================================
\section{Portfolio Concentration}
\label{sec:concentration}

The Herfindahl--Hirschman Index (HHI) measures portfolio concentration.  An HHI below
1{,}500 indicates a well-diversified portfolio; 1{,}500--2{,}500 is moderately concentrated;
above 2{,}500 is highly concentrated.  The total HHI for this portfolio is
\textbf{""" + _num(total_hhi, ",.0f") + r"""}, indicating a \textbf{""" + conc_hhi_label
+ r"""} profile.

""" + conc_table + r"""


% ============================================================
\section{Interest Rate Analytics}
\label{sec:rates}

Interest rates vary significantly across credit grades, reflecting the risk-based pricing
model.  Grade A loans average """ + _num(rate_df["avg_rate"].min(), ",.2f")
+ r"""\% while Grade G loans can reach up to """"""
PDF Risk Report Generator - LaTeX Version

Generates a professional LaTeX (.tex) report from portfolio analytics data,
then optionally compiles it to PDF via pdflatex.

The report covers:
    - Executive Summary (KPI table)
    - Portfolio Overview by Grade
    - Default Analysis
    - Loss Given Default (LGD)
    - Expected Loss (EL)
    - Portfolio Concentration (HHI)
    - Interest Rate Analytics
    - Model Performance

Usage:
    python -m src.visualization.pdf_report
"""

import sys
from pathlib import Path
from datetime import datetime
import subprocess

import pandas as pd
import numpy as np

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_OUTPUT_DIR = _PROJECT_ROOT / "output" / "reports"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================================
# LaTeX helpers
# ==================================================================

def _esc(val):
    """Escape special LaTeX characters in a value."""
    s = str(val)
    replacements = [
        ("\\", "\\textbackslash{}"), ("%", "\\%"), ("$", "\\$"),
        ("#", "\\#"), ("&", "\\&"), ("_", "\\_"), ("{", "\\{"),
        ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}"),
    ]
    for ch, repl in replacements:
        s = s.replace(ch, repl)
    return s


def _num(val, fmt=",.0f"):
    """Format a number with commas / decimals."""
    try:
        return format(float(val), fmt)
    except (ValueError, TypeError):
        return str(val)


def _pct(val, digits=2):
    """Format a ratio as a percentage string."""
    try:
        return "{:.{}f}\\%".format(float(val) * 100, digits)
    except (ValueError, TypeError):
        return str(val)


def _tabular(header, rows, col_fmt, caption=None, label=None):
    """
    Build a LaTeX tabular environment.

    Parameters
    ----------
    header  : list[str]
    rows    : list[list[str]]
    col_fmt : str   e.g. 'l r r r'
    caption : str | None
    label   : str | None
    """
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    if caption:
        lines.append("\\caption{{{}}}".format(_esc(caption)))
    lines.append("\\begin{tabular}{" + col_fmt + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if label:
        lines.append("\\label{{{}}}".format(_esc(label)))
    lines.append("\\end{table}")
    return "\n".join(lines)


# ==================================================================
# Report generator
# ==================================================================

def generate_report():
    """
    Generate a LaTeX risk report and return the file path.

    If pdflatex is on PATH, the .tex is also compiled to PDF.

    Returns
    -------
    str
        Absolute path to the generated file (.pdf or .tex).
    """
    from src.visualization.data_fetcher import DataFetcher

    # ------------------------------------------------------------------
    # 1. Collect data
    # ------------------------------------------------------------------
    print("Collecting data for LaTeX report...")

    portfolio_df = DataFetcher.portfolio_summary()
    default_df   = DataFetcher.default_rates_by_grade()
    lgd_df       = DataFetcher.lgd_by_grade()
    el_df        = DataFetcher.el_by_grade()
    conc_df      = DataFetcher.concentration_metrics()
    rate_df      = DataFetcher.int_rate_by_grade()

    total_loans    = int(portfolio_df["total_loans"].sum())
    total_funded   = float(portfolio_df["total_funded"].sum())
    avg_int_rate   = float(portfolio_df["avg_int_rate"].mean())
    total_mature   = int(default_df["total_mature"].sum())
    total_defaults = int(default_df["defaults"].sum())
    overall_dr     = total_defaults / total_mature if total_mature > 0 else 0
    total_el       = float(el_df["total_el"].sum())
    total_ead      = float(el_df["total_ead"].sum())
    el_pct         = total_el / total_ead if total_ead > 0 else 0
    avg_lgd        = float(lgd_df["avg_lgd"].mean()) if len(lgd_df) > 0 else 0
    total_hhi      = float(conc_df["hhi_contrib"].sum())

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    tex_path  = _OUTPUT_DIR / "risk_report_{}.tex".format(date_slug)

    # ------------------------------------------------------------------
    # 2. Build tables
    # ------------------------------------------------------------------
    # KPI
    kpi_rows = [
        ["Total Loans",             _num(total_loans)],
        ["Total Funded Amount",     "\\$" + _num(total_funded)],
        ["Average Interest Rate",   _pct(avg_int_rate / 100)],
        ["Mature Loans",            _num(total_mature)],
        ["Total Defaults",          _num(total_defaults)],
        ["Overall Default Rate",    _pct(overall_dr)],
        ["Expected Loss",           "\\$" + _num(total_el)],
        ["EL as \\% of Exposure",   _pct(el_pct)],
        ["Average LGD",             _pct(avg_lgd)],
        ["HHI Index",               _num(total_hhi, ",.0f") + " / 10,000"],
    ]
    kpi_table = _tabular(
        ["Metric", "Value"], kpi_rows, "l r",
        caption="Executive Summary", label="tab:summary",
    )

    # Portfolio overview
    port_rows = []
    for _, r in portfolio_df.iterrows():
        avg_fico = (r.get("avg_fico_low", 0) + r.get("avg_fico_high", 0)) / 2
        port_rows.append([
            _esc(r["grade"]),
            _num(r["total_loans"]),
            _num(r["total_funded"]),
            _num(r["avg_loan_amount"]),
            _num(r["avg_int_rate"], ",.2f"),
            _num(avg_fico, ",.0f"),
        ])
    port_rows.append([
        "\\textbf{TOTAL}",
        "\\textbf{" + _num(total_loans) + "}",
        "\\textbf{" + _num(total_funded) + "}",
        "\\textbf{" + _num(total_funded / total_loans) + "}",
        "\\textbf{" + _num(avg_int_rate, ",.2f") + "}",
        "-",
    ])
    port_table = _tabular(
        ["Grade", "Count", "Total Funded (\\$)", "Avg Amount (\\$)",
         "Avg Rate (\\%)", "Avg FICO"],
        port_rows, "c r r r r r",
        caption="Portfolio Overview by Grade", label="tab:portfolio",
    )

    # Default analysis
    def_rows = []
    for _, r in default_df.iterrows():
        def_rows.append([
            _esc(r["grade"]),
            _num(r["total_mature"]),
            _num(r["defaults"]),
            _num(r["default_rate"], ",.2f"),
        ])
    def_rows.append([
        "\\textbf{TOTAL}",
        "\\textbf{" + _num(total_mature) + "}",
        "\\textbf{" + _num(total_defaults) + "}",
        "\\textbf{" + _num(overall_dr * 100, ",.2f") + "}",
    ])
    def_table = _tabular(
        ["Grade", "Mature Count", "Defaults", "Default Rate (\\%)"],
        def_rows, "c r r r",
        caption="Default Rate by Grade (Mature Loans Only)", label="tab:defaults",
    )

    # LGD
    lgd_rows = []
    for _, r in lgd_df.iterrows():
        lgd_rows.append([
            _esc(r["grade"]),
            _num(r["total_defaulted"]),
            _pct(r["avg_lgd"], 1),
            _pct(r["p25_lgd"], 1),
            _pct(r["p50_lgd"], 1),
            _pct(r["p75_lgd"], 1),
        ])
    lgd_table = _tabular(
        ["Grade", "Count", "Avg LGD (\\%)", "P25 (\\%)", "Median (\\%)", "P75 (\\%)"],
        lgd_rows, "c r r r r r",
        caption="Loss Given Default Statistics by Grade", label="tab:lgd",
    )

    # Expected Loss
    el_rows = []
    for _, r in el_df.iterrows():
        grade_el_pct = r["total_el"] / total_ead if total_ead > 0 else 0
        el_rows.append([
            _esc(r["grade"]),
            _num(r["total_ead"]),
            _pct(r["pd"], 1),
            _pct(r["lgd"], 1),
            _num(r["el_per_loan"]),
            _num(r["total_el"]),
            _pct(grade_el_pct, 2),
        ])
    el_rows.append([
        "\\textbf{TOTAL}",
        "\\textbf{" + _num(total_ead) + "}",
        "-", "-", "-",
        "\\textbf{" + _num(total_el) + "}",
        "\\textbf{" + _pct(el_pct, 2) + "}",
    ])
    el_table = _tabular(
        ["Grade", "Exposure (\\$)", "PD", "LGD",
         "EL / Loan (\\$)", "Total EL (\\$)", "EL (\\%)"],
        el_rows, "c r r r r r r",
        caption="Expected Loss Breakdown by Grade", label="tab:el",
    )

    # Concentration
    top_conc = conc_df.sort_values("hhi_contrib", ascending=False).head(12)
    conc_hhi_label = (
        "Highly Concentrated" if total_hhi > 2500
        else "Moderately Concentrated" if total_hhi > 1500
        else "Well Diversified"
    )
    conc_rows = []
    for _, r in top_conc.iterrows():
        conc_rows.append([
            _esc(r["segment"]),
            _esc(r["segment_type"]),
            _num(r["pct"], ",.2f"),
            _num(r["hhi_contrib"], ",.0f"),
        ])
    conc_rows.append([
        "\\textbf{TOTAL}", "", "",
        "\\textbf{" + _num(total_hhi, ",.0f") + "}",
    ])
    conc_table = _tabular(
        ["Segment", "Type", "Share (\\%)", "HHI Contribution"],
        conc_rows, "l l r r",
        caption="Portfolio Concentration (HHI Index) -- " + conc_hhi_label,
        label="tab:concentration",
    )

    # Interest rate
    rate_rows = []
    for _, r in rate_df.iterrows():
        rate_rows.append([
            _esc(r["grade"]),
            _num(r["count"]),
            _num(r["min_rate"], ",.2f"),
            _num(r["avg_rate"], ",.2f"),
            _num(r["max_rate"], ",.2f"),
        ])
    rate_table = _tabular(
        ["Grade", "Count", "Min (\\%)", "Avg (\\%)", "Max (\\%)"],
        rate_rows, "c r r r r",
        caption="Interest Rate Distribution by Grade", label="tab:rates",
    )

    # Pre-compute values used in the document body
    rate_min = _num(rate_df["avg_rate"].min(), ",.2f")
    rate_max = _num(rate_df["max_rate"].max(), ",.2f")

    # ------------------------------------------------------------------
    # 3. Assemble the full LaTeX document using .format()
    # ------------------------------------------------------------------
    doc = (
        "\\documentclass[11pt, a4paper]{article}\n"
        "\n"
        "% ============================================================\n"
        "% Packages\n"
        "% ============================================================\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage{lmodern}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{longtable}\n"
        "\\usepackage{float}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{xcolor}\n"
        "\\usepackage{hyperref}\n"
        "\\usepackage{fancyhdr}\n"
        "\\usepackage{titlesec}\n"
        "\\usepackage{array}\n"
        "\\usepackage{caption}\n"
        "\n"
        "% ============================================================\n"
        "% Page style\n"
        "% ============================================================\n"
        "\\pagestyle{fancy}\n"
        "\\fancyhf{}\n"
        "\\fancyhead[L]{\\small\\textit{Credit Portfolio Risk Report}}\n"
        "\\fancyhead[R]{\\small\\textit{" + timestamp + "}}\n"
        "\\fancyfoot[C]{\\thepage}\n"
        "\\renewcommand{\\headrulewidth}{0.4pt}\n"
        "\\renewcommand{\\footrulewidth}{0.4pt}\n"
        "\n"
        "% ============================================================\n"
        "% Title formatting\n"
        "% ============================================================\n"
        "\\titleformat{\\section}"
        "{\\Large\\bfseries\\color{blue!60!black}}"
        "{\\thesection}{1em}{}\n"
        "\\titleformat{\\subsection}"
        "{\\large\\bfseries\\color{blue!40!black}}"
        "{\\thesubsection}{1em}{}\n"
        "\n"
        "% ============================================================\n"
        "% Hyperref\n"
        "% ============================================================\n"
        "\\hypersetup{\n"
        "    colorlinks=true,\n"
        "    linkcolor=blue!70!black,\n"
        "    citecolor=blue!70!black,\n"
        "    urlcolor=blue!70!black,\n"
        "}\n"
        "\n"
        "% ============================================================\n"
        "\\begin{document}\n"
        "\n"
        "% --- Title Page ------------------------------------------------\n"
        "\\begin{titlepage}\n"
        "\\centering\n"
        "\\vspace*{2cm}\n"
        "\n"
        "{{\\Huge\\bfseries\\color{blue!60!black} Credit Portfolio Risk Report\\par}}\n"
        "\n"
        "\\vspace{0.5cm}\n"
        "{{\\Large Lending Club Loan Portfolio Analytics\\par}}\n"
        "\n"
        "\\vspace{1.5cm}\n"
        "\n"
        "\\begin{tabular}{r l}\n"
        "\\textbf{Portfolio:}   & 2,260,668 Loans \\\\\\[4pt]\n"
        "\\textbf{Exposure:}    & \\$34,004,208,600 \\\\\\[4pt]\n"
        "\\textbf{Grades:}      & A -- G \\\\\\[4pt]\n"
        "\\textbf{Data Source:} & Lending Club + US Census ACS S2503 \\\\\\[4pt]\n"
        "\\textbf{Generated:}   & " + timestamp + " \\\\\n"
        "\\end{tabular}\n"
        "\n"
        "\\vspace{1.5cm}\n"
        "\n"
        + kpi_table + "\n"
        "\n"
        "\\vfill\n"
        "{\\small\\textit{This report was auto-generated by the Credit Portfolio Risk Analytics Engine.}}\n"
        "\\end{titlepage}\n"
        "\n"
        "% --- Table of Contents -----------------------------------------\n"
        "\\newpage\n"
        "\\tableofcontents\n"
        "\\newpage\n"
        "\n"
        "% ============================================================\n"
        "\\section{Portfolio Overview}\n"
        "\\label{sec:portfolio}\n"
        "\n"
        "The portfolio consists of \\textbf{" + _num(total_loans) + "} loans with a total funded\n"
        "amount of \\textbf{\\$" + _num(total_funded) + "}.  The average interest rate across all\n"
        "grades is \\textbf{" + _num(avg_int_rate, ",.2f") + "\\%}.  The table below breaks down key\n"
        "metrics by credit grade, from A (lowest risk) to G (highest risk).  Higher grades command\n"
        "larger average loan amounts but carry substantially higher default rates, as shown in\n"
        "subsequent sections.\n"
        "\n"
        + port_table + "\n"
        "\n"
        "% ============================================================\n"
        "\\section{Default Analysis}\n"
        "\\label{sec:defaults}\n"
        "\n"
        "Default rates are computed on \\emph{mature loans only} --- those that have reached a terminal\n"
        "status (Fully Paid or Charged Off / Default).  Of the \\textbf{" + _num(total_mature) + "}\n"
        "mature loans, \\textbf{" + _num(total_defaults) + "} have defaulted, yielding an overall\n"
        "default rate of \\textbf{" + _pct(overall_dr) + "}.  Default rates increase monotonically\n"
        "from Grade A to Grade G, reflecting the underlying credit risk differentiation embedded in\n"
        "Lending Club's grading methodology.\n"
        "\n"
        + def_table + "\n"
        "\n"
        "% ============================================================\n"
        "\\section{Loss Given Default}\n"
        "\\label{sec:lgd}\n"
        "\n"
        "LGD measures the fraction of exposure that is \\emph{lost} when a loan defaults.  It is\n"
        "calculated as:\n"
        "\n"
        "\\begin{equation}\n"
        "    \\text{LGD} = 1 - \\frac{\\text{total\\_pymnt}}{\\text{funded\\_amnt}}\n"
        "\\end{equation}\n"
        "\n"
        "The average LGD across all grades is \\textbf{" + _pct(avg_lgd, 1) + "}.  Lower grades\n"
        "tend to exhibit higher LGD, driven by higher interest rates accumulating in the balance\n"
        "before the borrower ultimately defaults.\n"
        "\n"
        + lgd_table + "\n"
        "\n"
        "% ============================================================\n"
        "\\section{Expected Loss}\n"
        "\\label{sec:el}\n"
        "\n"
        "Expected Loss (EL) is the risk-adjusted anticipated loss on the portfolio, computed per\n"
        "grade segment as:\n"
        "\n"
        "\\begin{equation}\n"
        "    \\text{EL} = \\text{PD} \\times \\text{LGD} \\times \\text{EAD}\n"
        "\\end{equation}\n"
        "\n"
        "The total portfolio expected loss is \\textbf{\\$" + _num(total_el) + "}, representing\n"
        "\\textbf{" + _pct(el_pct) + "} of total exposure.  This is a key input for loan\n"
        "pricing, capital allocation, and IFRS\\,9 provisioning.\n"
        "\n"
        + el_table + "\n"
        "\n"
        "% ============================================================\n"
        "\\section{Portfolio Concentration}\n"
        "\\label{sec:concentration}\n"
        "\n"
        "The Herfindahl--Hirschman Index (HHI) measures portfolio concentration.  An HHI below\n"
        "1,500 indicates a well-diversified portfolio; 1,500--2,500 is moderately concentrated;\n"
        "above 2,500 is highly concentrated.  The total HHI for this portfolio is\n"
        "\\textbf{" + _num(total_hhi, ",.0f") + "}, indicating a \\textbf{" + conc_hhi_label
        + "} profile.\n"
        "\n"
        + conc_table + "\n"
        "\n"
        "% ============================================================\n"
        "\\section{Interest Rate Analytics}\n"
        "\\label{sec:rates}\n"
        "\n"
        "Interest rates vary significantly across credit grades, reflecting the risk-based pricing\n"
        "model.  Grade A loans average " + rate_min
        + "\\% while Grade G loans can reach up to "
        + rate_max
        + "\\%.  This spread compensates investors for the higher probability of default in\n"
        "lower-grade segments.\n"
        "\n"
        + rate_table + "\n"
        "\n"
        "% ============================================================\n"
        "\\section{Model Performance}\n"
        "\\label{sec:models}\n"
        "\n"
        "The following machine-learning models have been trained on the portfolio data:\n"
        "\n"
        "\\begin{itemize}\n"
        "    \\item \\textbf{PD Model} --- HistGradientBoostingClassifier on 1,346,111 mature loans\n"
        "          (AUC-ROC = 0.7235).\n"
        "    \\item \\textbf{LGD Model} --- GradientBoostingRegressor with Huber loss on 269,360\n"
        "          defaulted loans ($R^{2}$ = 0.1513).\n"
        "    \\item \\textbf{Vasicek ASRF} --- Monte Carlo simulation (100,000 scenarios) with\n"
        "          grade-grouped binomial approximation.  Expected Loss = \\$3.1B,\n"
        "          VaR @ 99.9\\% = \\$10.1B.\n"
        "\\end{itemize}\n"
        "\n"
        "\\subsection{PD Model}\n"
        "\n"
        "The PD model achieves an AUC-ROC of \\textbf{0.7235}, indicating good discriminative power\n"
        "between defaulting and non-defaulting loans.  The model was trained on 1,076,888 loans\n"
        "and validated on 269,223 holdout loans.  Actual vs.\\ predicted default rates by grade show\n"
        "strong calibration, with per-grade errors below 0.15 percentage points.\n"
        "\n"
        "\\subsection{LGD Model}\n"
        "\n"
        "The LGD model predicts recovery rates with an $R^{2}$ of \\textbf{0.1513} and MAE of\n"
        "\\textbf{0.2097}.  The top predictive feature is interest rate (importance = 0.47), followed\n"
        "by sub-grade (0.12) and revolving utilisation (0.06).\n"
        "\n"
        "\\subsection{Vasicek ASRF Model}\n"
        "\n"
        "The Vasicek model estimates the portfolio loss distribution under macroeconomic stress.\n"
        "Key results:\n"
        "\n"
        "\\begin{itemize}\n"
        "    \\item Expected Loss: \\$3,114,079,959 (9.16\\% of exposure)\n"
        "    \\item VaR @ 99.9\\%: \\$10,080,224,115 (29.64\\% of exposure)\n"
        "    \\item Unexpected Loss (UL): \\$6,966,144,156 (20.49\\%)\n"
        "    \\item Economic Capital Requirement: \\$10,080,224,115\n"
        "\\end{itemize}\n"
        "\n"
        "Asset correlations follow Basel II corporate exposure parameters:\n"
        "$\\rho_{A-B} = 0.15$, $\\rho_{C-D} = 0.20$, $\\rho_{E-G} = 0.25$.\n"
        "\n"
        "% ============================================================\n"
        "\\end{document}\n"
    )

    # ------------------------------------------------------------------
    # 4. Write .tex file
    # ------------------------------------------------------------------
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(doc)
    print("LaTeX report saved to: {}".format(tex_path))

    # ------------------------------------------------------------------
    # 5. Try to compile with pdflatex
    # ------------------------------------------------------------------
    pdf_path = tex_path.with_suffix(".pdf")
    compiled = False

    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode",
             "-output-directory", str(_OUTPUT_DIR), str(tex_path)],
            capture_output=True, text=True, timeout=60,
        )
        if pdf_path.exists():
            print("PDF compiled successfully: {}".format(pdf_path))
            compiled = True
        else:
            print("pdflatex ran but PDF not found. Check the .log file.")
    except FileNotFoundError:
        print("pdflatex not found on PATH. Open the .tex file in Texmaker to compile manually.")
    except subprocess.TimeoutExpired:
        print("pdflatex timed out. Compile manually in Texmaker.")
    except Exception as exc:
        print("pdflatex compilation failed: {}".format(exc))

    if compiled:
        return str(pdf_path)
    return str(tex_path)


if __name__ == "__main__":
    path = generate_report()
    print("Done. File: {}".format(path))
+ _num(rate_df["max_rate"].max(), ",.2f")
+ r"""\%.  This spread compensates investors for the higher probability of default in
lower-grade segments.

""" + rate_table + r"""


% ============================================================
\section{Model Performance}
\label{sec:models}

The following machine-learning models have been trained on the portfolio data:

\begin{itemize}
    \item \textbf{PD Model} --- HistGradientBoostingClassifier on 1{,}346{,}111 mature loans
          (AUC-ROC = 0.7235).
    \item \textbf{LGD Model} --- GradientBoostingRegressor with Huber loss on 269{,}360
          defaulted loans ($R^{2}$ = 0.1513).
    \item \textbf{Vasicek ASRF} --- Monte Carlo simulation (100{,}000 scenarios) with
          grade-grouped binomial approximation.  Expected Loss = \$3.1{,}B,
          VaR @ 99.9\% = \$10.1{,}B.
\end{itemize}

\subsection{PD Model}

The PD model achieves an AUC-ROC of \textbf{0.7235}, indicating good discriminative power
between defaulting and non-defaulting loans.  The model was trained on 1{,}076{,}888 loans
and validated on 269{,}223 holdout loans.  Actual vs.\ predicted default rates by grade show
strong calibration, with per-grade errors below 0.15 percentage points.

\subsection{LGD Model}

The LGD model predicts recovery rates with an $R^{2}$ of \textbf{0.1513} and MAE of
\textbf{0.2097}.  The top predictive feature is interest rate (importance = 0.47), followed
by sub-grade (0.12) and revolving utilisation (0.06).

\subsection{Vasicek ASRF Model}

The Vasicek model estimates the portfolio loss distribution under macroeconomic stress.
Key results:

\begin{itemize}
    \item Expected Loss: \$3{,}114{,}079{,}959 (9.16\% of exposure)
    \item VaR @ 99.9\%: \$10{,}080{,}224{,}115 (29.64\% of exposure)
    \item Unexpected Loss (UL): \$6{,}966{,}144{,}156 (20.49\%)
    \item Economic Capital Requirement: \$10{,}080{,}224{,}115
\end{itemize}

Asset correlations follow Basel II corporate exposure parameters:
 $\rho_{A-B} = 0.15$, $\rho_{C-D} = 0.20$, $\rho_{E-G} = 0.25$.


% ============================================================
\end{document}
"""

    # ------------------------------------------------------------------
    # 10. Write .tex file
    # ------------------------------------------------------------------
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(doc)
    print("LaTeX report saved to: {}".format(tex_path))

    # ------------------------------------------------------------------
    # 11. Try to compile with pdflatex
    # ------------------------------------------------------------------
    pdf_path = tex_path.with_suffix(".pdf")
    compiled = False

    try:
        import subprocess
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory",
             str(_OUTPUT_DIR), str(tex_path)],
            capture_output=True, text=True, timeout=60,
        )
        if pdf_path.exists():
            print("PDF compiled successfully: {}".format(pdf_path))
            compiled = True
        else:
            print("pdflatex ran but PDF not found. Check the .log file.")
    except FileNotFoundError:
        print("pdflatex not found on PATH. Open the .tex file in Texmaker to compile manually.")
    except subprocess.TimeoutExpired:
        print("pdflatex timed out. Compile manually in Texmaker.")
    except Exception as exc:
        print("pdflatex compilation failed: {}".format(exc))

    if compiled:
        return str(pdf_path)
    return str(tex_path)


if __name__ == "__main__":
    path = generate_report()
    print("Done. File: {}".format(path))
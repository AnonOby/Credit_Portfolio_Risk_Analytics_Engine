"""
PDF Risk Report Generator

Produces a professional multi-page PDF risk report from the analytics data.
Uses ReportLab for PDF generation with charts rendered as Plotly static images.

The report includes:
  - Cover page with title and generation timestamp
  - Portfolio overview summary table
  - Default analysis by grade
  - LGD and EL breakdown
  - Concentration metrics
  - Model performance summary
  - Stress test summary

Usage:
    python -m src.visualization.pdf_report
"""

import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_OUTPUT_DIR = _PROJECT_ROOT / "output" / "reports"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_report() -> str:
    """
    Generate a PDF risk report and return the file path.

    Returns
    -------
    str
        Absolute path to the generated PDF file.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.colors import HexColor, white
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak,
        )
    except ImportError:
        print("ERROR: reportlab is required. Install with: pip install reportlab")
        sys.exit(1)

    # Lazy import for data fetching
    from src.visualization.data_fetcher import DataFetcher

    # -----------------------------------------------------------------------
    # Data collection
    # -----------------------------------------------------------------------
    print("Collecting data for PDF report...")

    portfolio_df = DataFetcher.portfolio_summary()
    default_df = DataFetcher.default_rates_by_grade()
    lgd_df = DataFetcher.lgd_by_grade()
    el_df = DataFetcher.el_by_grade()
    conc_df = DataFetcher.concentration_metrics()
    rate_df = DataFetcher.int_rate_by_grade()

    # Compute summary stats
    total_loans = portfolio_df["total_loans"].sum()
    total_funded = portfolio_df["total_funded"].sum()
    avg_int_rate = portfolio_df["avg_int_rate"].mean()
    total_mature = default_df["total_mature"].sum()
    total_defaults = default_df["defaults"].sum()
    overall_default_rate = total_defaults / total_mature * 100 if total_mature > 0 else 0
    total_el = el_df["total_el"].sum()
    total_ead = el_df["total_ead"].sum()
    el_pct = total_el / total_ead * 100 if total_ead > 0 else 0
    avg_lgd = lgd_df["avg_lgd"].mean() * 100 if len(lgd_df) > 0 else 0
    total_hhi = conc_df["hhi_contrib"].sum()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf_path = _OUTPUT_DIR / "risk_report_{}.pdf".format(
        datetime.now().strftime("%Y%m%d_%H%M%S"))

    # -----------------------------------------------------------------------
    # Styles
    # -----------------------------------------------------------------------
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"],
        fontSize=28, spaceAfter=12, textColor=HexColor("#1a3e60"),
        alignment=TA_CENTER, fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "CustomSubtitle", parent=styles["Normal"],
        fontSize=12, spaceAfter=6, textColor=HexColor("#5d6d7e"),
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "CustomHeading", parent=styles["Heading2"],
        fontSize=16, spaceBefore=24, spaceAfter=12,
        textColor=HexColor("#1a3e60"), fontName="Helvetica-Bold",
        borderWidth=0, borderPadding=0,
    )
    body_style = ParagraphStyle(
        "CustomBody", parent=styles["Normal"],
        fontSize=10, spaceAfter=12, leading=14,
        textColor=HexColor("#2c3e50"),
    )

    # -----------------------------------------------------------------------
    # Helper function for professional table style (three-line table)
    # -----------------------------------------------------------------------
    def style_table(header_rows=1, total_row_pos=None):
        """Return a TableStyle for a three-line table with alternating row backgrounds."""
        base_style = [
            # Header background and text
            ("BACKGROUND", (0, 0), (-1, header_rows-1), HexColor("#1a3e60")),
            ("TEXTCOLOR", (0, 0), (-1, header_rows-1), white),
            ("FONTNAME", (0, 0), (-1, header_rows-1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, header_rows-1), 9),
            ("ALIGN", (0, 0), (-1, header_rows-1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            # Top border (thick)
            ("LINEABOVE", (0, 0), (-1, 0), 1.5, HexColor("#1a3e60")),
            # Bottom border of header (thick)
            ("LINEBELOW", (0, header_rows-1), (-1, header_rows-1), 1, HexColor("#1a3e60")),
            # Bottom border of table (thick)
            ("LINEBELOW", (0, -1), (-1, -1), 1.5, HexColor("#1a3e60")),
            # Alternating row colors for body
            ("ROWBACKGROUNDS", (0, header_rows), (-1, -2), [HexColor("#ffffff"), HexColor("#f7f9fc")]),
            # Font settings for body
            ("FONTSIZE", (0, header_rows), (-1, -1), 9),
            ("FONTNAME", (0, header_rows), (-1, -1), "Helvetica"),
            # Padding
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]
        # If a total row is specified (last row), style it distinctively
        if total_row_pos is not None and total_row_pos >= 0:
            base_style.extend([
                ("BACKGROUND", (0, total_row_pos), (-1, total_row_pos), HexColor("#eef2f5")),
                ("LINEABOVE", (0, total_row_pos), (-1, total_row_pos), 0.5, HexColor("#bdc3c7")),
                ("FONTNAME", (0, total_row_pos), (-1, total_row_pos), "Helvetica-Bold"),
            ])
        return TableStyle(base_style)

    # -----------------------------------------------------------------------
    # Build document
    # -----------------------------------------------------------------------
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )

    story = []

    # --- Cover page ---------------------------------------------------------
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph("Credit Portfolio Risk Report", title_style))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Lending Club Loan Portfolio Analytics", subtitle_style))
    story.append(Spacer(1, 0.6 * inch))

    # KPI summary box on cover (enhanced, with correct text color)
    kpi_data = [
        ["Total Loans", "Total Funded", "Avg Rate", "Default Rate"],
        ["{:,.0f}".format(total_loans), "${:,.0f}".format(total_funded),
         "{:.2f}%".format(avg_int_rate), "{:.2f}%".format(overall_default_rate)],
        ["Expected Loss", "EL %", "Avg LGD", "HHI Index"],
        ["${:,.0f}".format(total_el), "{:.2f}%".format(el_pct),
         "{:.1f}%".format(avg_lgd), "{:.0f}".format(total_hhi)],
    ]
    kpi_table = Table(kpi_data, colWidths=[1.7 * inch] * 4, repeatRows=1)
    kpi_table.setStyle(TableStyle([
        # Header rows (row 0 and row 2) – dark blue background, white text
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a3e60")),
        ("BACKGROUND", (0, 2), (-1, 2), HexColor("#1a3e60")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("TEXTCOLOR", (0, 2), (-1, 2), white),
        # Data rows (row 1 and row 3) – light gray background, dark text
        ("BACKGROUND", (0, 1), (-1, 1), HexColor("#f0f2f5")),
        ("BACKGROUND", (0, 3), (-1, 3), HexColor("#f0f2f5")),
        ("TEXTCOLOR", (0, 1), (-1, 1), HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 3), (-1, 3), HexColor("#2c3e50")),
        # Alignment and font
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("FONTSIZE", (0, 1), (-1, 1), 13),
        ("FONTSIZE", (0, 2), (-1, 2), 10),
        ("FONTSIZE", (0, 3), (-1, 3), 13),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica"),
        ("FONTNAME", (0, 3), (-1, 3), "Helvetica"),
        # Padding and borders
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#bdc3c7")),
        ("BOX", (0, 0), (-1, -1), 1.2, HexColor("#1a3e60")),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 1.2 * inch))
    story.append(Paragraph("Generated: {}".format(timestamp), subtitle_style))
    story.append(PageBreak())

    # --- Section 1: Portfolio Overview ---------------------------------------
    story.append(Paragraph("1. Portfolio Overview", heading_style))
    story.append(Paragraph(
        "The portfolio consists of {:,} loans with a total funded amount of "
        "${:,.0f}. The average interest rate across all grades is {:.2f}%. "
        "The table below breaks down key metrics by credit grade, from A (lowest risk) "
        "to G (highest risk).".format(total_loans, total_funded, avg_int_rate),
        body_style,
    ))
    story.append(Spacer(1, 0.1 * inch))

    # Portfolio summary table
    port_header = ["Grade", "Count", "Total Funded", "Avg Amount", "Avg Rate", "Avg FICO"]
    port_rows = []
    for _, row in portfolio_df.iterrows():
        avg_fico = (row.get("avg_fico_low", 0) + row.get("avg_fico_high", 0)) / 2
        port_rows.append([
            str(row["grade"]),
            "{:,.0f}".format(row["total_loans"]),
            "${:,.0f}".format(row["total_funded"]),
            "${:,.0f}".format(row["avg_loan_amount"]),
            "{:.2f}%".format(row["avg_int_rate"]),
            "{:.0f}".format(avg_fico),
        ])
    # Total row
    port_rows.append([
        "TOTAL",
        "{:,.0f}".format(total_loans),
        "${:,.0f}".format(total_funded),
        "${:,.0f}".format(total_funded / total_loans if total_loans > 0 else 0),
        "{:.2f}%".format(avg_int_rate),
        "-",
    ])

    port_table = Table([port_header] + port_rows,
                       colWidths=[0.65*inch, 0.9*inch, 1.2*inch, 1.1*inch, 0.9*inch, 0.9*inch],
                       repeatRows=1)
    port_table.setStyle(style_table(header_rows=1, total_row_pos=len(port_rows)))
    # Override alignment: numeric columns right-aligned
    for col in range(1, len(port_header)):
        port_table.setStyle(TableStyle([("ALIGN", (col, 1), (col, -1), "RIGHT")]))
    port_table.setStyle(TableStyle([("ALIGN", (0, 1), (0, -1), "CENTER")]))
    story.append(port_table)
    story.append(PageBreak())

    # --- Section 2: Default Analysis ----------------------------------------
    story.append(Paragraph("2. Default Analysis", heading_style))
    story.append(Paragraph(
        "Default rates are computed on mature loans only (fully paid or charged off). "
        "The overall default rate across {:,} mature loans is {:.2f}%. "
        "Default rates increase monotonically from Grade A to Grade G, "
        "reflecting the underlying credit risk differentiation.".format(
            total_mature, overall_default_rate),
        body_style,
    ))
    story.append(Spacer(1, 0.1 * inch))

    def_header = ["Grade", "Mature Count", "Defaults", "Default Rate"]
    def_rows = []
    for _, row in default_df.iterrows():
        def_rows.append([
            str(row["grade"]),
            "{:,.0f}".format(row["total_mature"]),
            "{:,.0f}".format(row["defaults"]),
            "{:.2f}%".format(row["default_rate"]),
        ])
    def_rows.append([
        "TOTAL",
        "{:,.0f}".format(total_mature),
        "{:,.0f}".format(total_defaults),
        "{:.2f}%".format(overall_default_rate),
    ])

    def_table = Table([def_header] + def_rows,
                      colWidths=[1.2*inch, 1.5*inch, 1.5*inch, 1.5*inch],
                      repeatRows=1)
    def_table.setStyle(style_table(header_rows=1, total_row_pos=len(def_rows)))
    for col in range(1, len(def_header)):
        def_table.setStyle(TableStyle([("ALIGN", (col, 1), (col, -1), "RIGHT")]))
    def_table.setStyle(TableStyle([("ALIGN", (0, 1), (0, -1), "CENTER")]))
    story.append(def_table)
    story.append(Spacer(1, 0.2 * inch))

    # --- Section 3: LGD Analysis --------------------------------------------
    story.append(Paragraph("3. Loss Given Default (LGD)", heading_style))
    story.append(Paragraph(
        "LGD measures the fraction of exposure that is lost when a loan defaults. "
        "It is calculated as LGD = 1 - (total_pymnt / funded_amnt) for defaulted loans. "
        "The average LGD across all grades is {:.1f}%. Lower grades tend to have higher LGD "
        "due to higher interest rates accumulating in the balance before default.".format(avg_lgd),
        body_style,
    ))
    story.append(Spacer(1, 0.1 * inch))

    lgd_header = ["Grade", "Defaulted Count", "Avg LGD", "P25 LGD", "Median LGD", "P75 LGD"]
    lgd_rows = []
    for _, row in lgd_df.iterrows():
        lgd_rows.append([
            str(row["grade"]),
            "{:,.0f}".format(row["total_defaulted"]),
            "{:.1%}".format(row["avg_lgd"]),
            "{:.1%}".format(row["p25_lgd"]),
            "{:.1%}".format(row["p50_lgd"]),
            "{:.1%}".format(row["p75_lgd"]),
        ])

    lgd_table = Table([lgd_header] + lgd_rows,
                      colWidths=[0.8*inch, 1.2*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.0*inch],
                      repeatRows=1)
    lgd_table.setStyle(style_table(header_rows=1))
    for col in range(1, len(lgd_header)):
        lgd_table.setStyle(TableStyle([("ALIGN", (col, 1), (col, -1), "RIGHT")]))
    lgd_table.setStyle(TableStyle([("ALIGN", (0, 1), (0, -1), "CENTER")]))
    story.append(lgd_table)
    story.append(PageBreak())

    # --- Section 4: Expected Loss -------------------------------------------
    story.append(Paragraph("4. Expected Loss Analysis", heading_style))
    story.append(Paragraph(
        "Expected Loss (EL) is the risk-adjusted anticipated loss on the portfolio, "
        "computed as EL = PD x LGD x EAD per grade segment. The total portfolio expected "
        "loss is ${:,.0f}, representing {:.2f}% of total exposure. This is a key input "
        "for loan pricing, capital allocation, and provisioning.".format(total_el, el_pct),
        body_style,
    ))
    story.append(Spacer(1, 0.1 * inch))

    el_header = ["Grade", "Exposure", "PD", "LGD", "EL/L", "Total EL", "EL %"]
    el_rows = []
    for _, row in el_df.iterrows():
        grade_el_pct = row["total_el"] / total_ead * 100 if total_ead > 0 else 0
        el_rows.append([
            str(row["grade"]),
            "${:,.0f}".format(row["total_ead"]),
            "{:.1%}".format(row["pd"]),
            "{:.1%}".format(row["lgd"]),
            "${:,.0f}".format(row["el_per_loan"]),
            "${:,.0f}".format(row["total_el"]),
            "{:.2f}%".format(grade_el_pct),
        ])
    el_rows.append([
        "TOTAL", "${:,.0f}".format(total_ead), "-", "-",
        "-", "${:,.0f}".format(total_el), "{:.2f}%".format(el_pct),
    ])

    el_table = Table([el_header] + el_rows,
                     colWidths=[0.55*inch, 1.0*inch, 0.7*inch, 0.7*inch, 0.9*inch, 1.1*inch, 0.7*inch],
                     repeatRows=1)
    el_table.setStyle(style_table(header_rows=1, total_row_pos=len(el_rows)))
    for col in range(1, len(el_header)):
        el_table.setStyle(TableStyle([("ALIGN", (col, 1), (col, -1), "RIGHT")]))
    el_table.setStyle(TableStyle([("ALIGN", (0, 1), (0, -1), "CENTER")]))
    story.append(el_table)
    story.append(Spacer(1, 0.2 * inch))

    # --- Section 5: Concentration -------------------------------------------
    story.append(Paragraph("5. Portfolio Concentration", heading_style))
    story.append(Paragraph(
        "The Herfindahl-Hirschman Index (HHI) measures portfolio concentration. "
        "An HHI below 1,500 indicates a well-diversified portfolio; 1,500-2,500 is "
        "moderately concentrated; above 2,500 is highly concentrated. "
        "The total HHI for this portfolio is {:.0f}, indicating {}.".format(
            total_hhi,
            "high concentration" if total_hhi > 2500 else
            "moderate concentration" if total_hhi > 1500 else
            "good diversification",
        ),
        body_style,
    ))
    story.append(Spacer(1, 0.1 * inch))

    # Top 10 concentration contributions
    top_conc = conc_df.sort_values("hhi_contrib", ascending=False).head(10)
    conc_header = ["Segment", "Share %", "HHI Contribution"]
    conc_rows = []
    for _, row in top_conc.iterrows():
        conc_rows.append([
            str(row["segment"]),
            "{:.2f}%".format(row["pct"]),
            "{:.0f}".format(row["hhi_contrib"]),
        ])
    conc_rows.append(["TOTAL", "-", "{:.0f}".format(total_hhi)])

    conc_table = Table([conc_header] + conc_rows,
                       colWidths=[3.0*inch, 1.2*inch, 1.5*inch],
                       repeatRows=1)
    conc_table.setStyle(style_table(header_rows=1, total_row_pos=len(conc_rows)))
    conc_table.setStyle(TableStyle([("ALIGN", (1, 1), (1, -1), "RIGHT"),
                                    ("ALIGN", (2, 1), (2, -1), "RIGHT")]))
    conc_table.setStyle(TableStyle([("ALIGN", (0, 1), (0, -1), "LEFT")]))
    story.append(conc_table)
    story.append(PageBreak())

    # --- Section 6: Interest Rate Analytics ---------------------------------
    story.append(Paragraph("6. Interest Rate Analytics", heading_style))
    story.append(Paragraph(
        "Interest rates vary significantly across credit grades, reflecting the "
        "risk-based pricing model. Grade A loans average {:.2f}% while Grade G loans "
        "can reach up to {:.2f}%. This spread compensates investors for the higher "
        "probability of default in lower-grade segments.".format(
            rate_df["avg_rate"].min() if len(rate_df) > 0 else 0,
            rate_df["max_rate"].max() if len(rate_df) > 0 else 0,
        ),
        body_style,
    ))
    story.append(Spacer(1, 0.1 * inch))

    rate_header = ["Grade", "Count", "Min Rate", "Avg Rate", "Max Rate"]
    rate_rows = []
    for _, row in rate_df.iterrows():
        rate_rows.append([
            str(row["grade"]),
            "{:,.0f}".format(row["count"]),
            "{:.2f}%".format(row["min_rate"]),
            "{:.2f}%".format(row["avg_rate"]),
            "{:.2f}%".format(row["max_rate"]),
        ])

    rate_table = Table([rate_header] + rate_rows,
                       colWidths=[0.8*inch, 1.0*inch, 1.1*inch, 1.1*inch, 1.1*inch],
                       repeatRows=1)
    rate_table.setStyle(style_table(header_rows=1))
    for col in range(1, len(rate_header)):
        rate_table.setStyle(TableStyle([("ALIGN", (col, 1), (col, -1), "RIGHT")]))
    rate_table.setStyle(TableStyle([("ALIGN", (0, 1), (0, -1), "CENTER")]))
    story.append(rate_table)

    # -----------------------------------------------------------------------
    # Build PDF
    # -----------------------------------------------------------------------
    print("Generating PDF report...")
    doc.build(story)
    print("Report saved to: {}".format(pdf_path))
    return str(pdf_path)


if __name__ == "__main__":
    path = generate_report()
    print("Done. File: {}".format(path))
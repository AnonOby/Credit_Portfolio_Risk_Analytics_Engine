"""
powerbi_export.py - Comprehensive Power BI Data Export
=======================================================

Exports 20+ CSV tables to data/powerbi/ for Power BI dashboard ingestion.

Each table targets a specific Power BI visual or analysis:
    - Portfolio overview by grade
    - Loan status distribution
    - Default analysis by grade, term, purpose, home ownership
    - LGD statistics by grade and term
    - Expected loss breakdown
    - Concentration metrics (grade, state, purpose, term)
    - Interest rate analytics
    - FICO score distribution
    - DTI analysis
    - Time-series issuance trends
    - Census enrichment data
    - Model performance metrics
    - Vasicek simulation results

All queries use PostgreSQL-compatible syntax with ::numeric casts.

Usage:
    python -m src.visualization.powerbi_export
    # Or import and call:
    from src.visualization.powerbi_export import export_powerbi_data
    export_powerbi_data()
"""

import pandas as pd
import sys
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.database.connection import get_engine
import config


# ==================================================================
# Output directory
# ==================================================================

_OUTPUT_DIR = config.BASE_DIR / "data" / "powerbi"


# ==================================================================
# SQL Query Definitions
# ==================================================================

def _q_portfolio_by_grade():
    """Portfolio summary by credit grade."""
    return """
        SELECT
            grade,
            COUNT(*)::numeric                                    AS total_loans,
            ROUND(SUM(loan_amnt)::numeric, 2)                   AS total_loan_amount,
            ROUND(AVG(loan_amnt)::numeric, 2)                   AS avg_loan_amount,
            ROUND(AVG(funded_amnt)::numeric, 2)                 AS avg_funded_amount,
            ROUND(AVG(int_rate)::numeric, 2)                    AS avg_int_rate,
            ROUND(AVG((fico_range_low + fico_range_high) / 2.0)::numeric, 1) AS avg_fico,
            ROUND(AVG(annual_inc)::numeric, 2)                  AS avg_annual_income,
            ROUND(AVG(dti)::numeric, 2)                         AS avg_dti,
            ROUND(AVG(emp_length)::numeric, 1)                  AS avg_emp_length,
            ROUND(AVG(credit_history_months)::numeric, 1)       AS avg_credit_history_months
        FROM loans_master
        GROUP BY grade
        ORDER BY grade
    """


def _q_portfolio_by_term():
    """Portfolio summary by loan term."""
    return """
        SELECT
            term,
            COUNT(*)::numeric                                    AS total_loans,
            ROUND(SUM(loan_amnt)::numeric, 2)                   AS total_loan_amount,
            ROUND(AVG(loan_amnt)::numeric, 2)                   AS avg_loan_amount,
            ROUND(AVG(int_rate)::numeric, 2)                    AS avg_int_rate,
            ROUND(AVG((fico_range_low + fico_range_high) / 2.0)::numeric, 1) AS avg_fico,
            ROUND(AVG(annual_inc)::numeric, 2)                  AS avg_annual_income,
            ROUND(AVG(dti)::numeric, 2)                         AS avg_dti
        FROM loans_master
        GROUP BY term
        ORDER BY term
    """


def _q_portfolio_by_grade_term():
    """Portfolio summary by grade and term cross-tabulation."""
    return """
        SELECT
            grade,
            term,
            COUNT(*)::numeric                                    AS total_loans,
            ROUND(SUM(loan_amnt)::numeric, 2)                   AS total_loan_amount,
            ROUND(AVG(loan_amnt)::numeric, 2)                   AS avg_loan_amount,
            ROUND(AVG(int_rate)::numeric, 2)                    AS avg_int_rate
        FROM loans_master
        GROUP BY grade, term
        ORDER BY grade, term
    """


def _q_loan_status():
    """Loan status distribution."""
    return """
        SELECT
            loan_status,
            COUNT(*)::numeric                                    AS total_loans,
            ROUND(SUM(loan_amnt)::numeric, 2)                   AS total_loan_amount,
            ROUND(100.0 * COUNT(*)::numeric / SUM(COUNT(*)) OVER (), 2) AS pct_of_portfolio
        FROM loans_master
        GROUP BY loan_status
        ORDER BY total_loans DESC
    """


def _q_default_rates_by_grade():
    """Default rate analysis by grade (mature loans only)."""
    return """
        WITH mature AS (
            SELECT *
            FROM loans_master
            WHERE loan_status IN (
                'Fully Paid', 'Charged Off', 'Default',
                'Does not meet the credit policy. Status:Fully Paid',
                'Does not meet the credit policy. Status:Charged Off'
            )
        ),
        labeled AS (
            SELECT *,
                CASE WHEN loan_status IN (
                    'Charged Off', 'Default',
                    'Does not meet the credit policy. Status:Charged Off'
                ) THEN 1 ELSE 0 END AS is_default
            FROM mature
        )
        SELECT
            grade,
            COUNT(*)::numeric                                    AS total_mature,
            SUM(is_default)::numeric                            AS total_defaults,
            ROUND(100.0 * SUM(is_default)::numeric / COUNT(*), 2) AS default_rate_pct,
            ROUND(AVG(loan_amnt)::numeric, 2)                   AS avg_loan_amount,
            ROUND(AVG(int_rate)::numeric, 2)                    AS avg_int_rate,
            ROUND(AVG((fico_range_low + fico_range_high) / 2.0)::numeric, 1) AS avg_fico,
            ROUND(AVG(annual_inc)::numeric, 2)                  AS avg_annual_income,
            ROUND(AVG(dti)::numeric, 2)                         AS avg_dti
        FROM labeled
        GROUP BY grade
        ORDER BY grade
    """


def _q_default_rates_by_term():
    """Default rate analysis by term."""
    return """
        WITH mature AS (
            SELECT *
            FROM loans_master
            WHERE loan_status IN (
                'Fully Paid', 'Charged Off', 'Default',
                'Does not meet the credit policy. Status:Fully Paid',
                'Does not meet the credit policy. Status:Charged Off'
            )
        ),
        labeled AS (
            SELECT *,
                CASE WHEN loan_status IN (
                    'Charged Off', 'Default',
                    'Does not meet the credit policy. Status:Charged Off'
                ) THEN 1 ELSE 0 END AS is_default
            FROM mature
        )
        SELECT
            term,
            COUNT(*)::numeric                                    AS total_mature,
            SUM(is_default)::numeric                            AS total_defaults,
            ROUND(100.0 * SUM(is_default)::numeric / COUNT(*), 2) AS default_rate_pct,
            ROUND(AVG(loan_amnt)::numeric, 2)                   AS avg_loan_amount,
            ROUND(AVG(int_rate)::numeric, 2)                    AS avg_int_rate
        FROM labeled
        GROUP BY term
        ORDER BY term
    """


def _q_default_rates_by_purpose():
    """Default rate analysis by loan purpose."""
    return """
        WITH mature AS (
            SELECT *
            FROM loans_master
            WHERE loan_status IN (
                'Fully Paid', 'Charged Off', 'Default',
                'Does not meet the credit policy. Status:Fully Paid',
                'Does not meet the credit policy. Status:Charged Off'
            )
        ),
        labeled AS (
            SELECT *,
                CASE WHEN loan_status IN (
                    'Charged Off', 'Default',
                    'Does not meet the credit policy. Status:Charged Off'
                ) THEN 1 ELSE 0 END AS is_default
            FROM mature
        )
        SELECT
            purpose,
            COUNT(*)::numeric                                    AS total_mature,
            SUM(is_default)::numeric                            AS total_defaults,
            ROUND(100.0 * SUM(is_default)::numeric / COUNT(*), 2) AS default_rate_pct,
            ROUND(AVG(loan_amnt)::numeric, 2)                   AS avg_loan_amount,
            ROUND(AVG(int_rate)::numeric, 2)                    AS avg_int_rate,
            ROUND(AVG(dti)::numeric, 2)                         AS avg_dti
        FROM labeled
        GROUP BY purpose
        ORDER BY total_mature DESC
    """


def _q_default_rates_by_home_ownership():
    """Default rate analysis by home ownership status."""
    return """
        WITH mature AS (
            SELECT *
            FROM loans_master
            WHERE loan_status IN (
                'Fully Paid', 'Charged Off', 'Default',
                'Does not meet the credit policy. Status:Fully Paid',
                'Does not meet the credit policy. Status:Charged Off'
            )
        ),
        labeled AS (
            SELECT *,
                CASE WHEN loan_status IN (
                    'Charged Off', 'Default',
                    'Does not meet the credit policy. Status:Charged Off'
                ) THEN 1 ELSE 0 END AS is_default
            FROM mature
        )
        SELECT
            home_ownership,
            COUNT(*)::numeric                                    AS total_mature,
            SUM(is_default)::numeric                            AS total_defaults,
            ROUND(100.0 * SUM(is_default)::numeric / COUNT(*), 2) AS default_rate_pct,
            ROUND(AVG(annual_inc)::numeric, 2)                  AS avg_annual_income,
            ROUND(AVG(dti)::numeric, 2)                         AS avg_dti
        FROM labeled
        GROUP BY home_ownership
        ORDER BY total_mature DESC
    """


def _q_lgd_by_grade():
    """LGD statistics by grade."""
    return """
        SELECT
            grade,
            COUNT(*)::numeric                                    AS total_defaulted,
            ROUND(AVG(1 - (total_pymnt / funded_amnt))::numeric, 4) AS avg_lgd,
            ROUND(AVG(total_pymnt / funded_amnt)::numeric, 4)   AS avg_recovery_rate,
            ROUND(AVG(funded_amnt)::numeric, 2)                 AS avg_exposure,
            ROUND(AVG(funded_amnt - total_pymnt)::numeric, 2)   AS avg_loss_amount,
            ROUND(SUM(funded_amnt - total_pymnt)::numeric, 2)   AS total_loss_amount,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (
                ORDER BY 1 - total_pymnt / funded_amnt)::numeric, 4) AS median_lgd,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (
                ORDER BY 1 - total_pymnt / funded_amnt)::numeric, 4) AS p25_lgd,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (
                ORDER BY 1 - total_pymnt / funded_amnt)::numeric, 4) AS p75_lgd
        FROM loans_master
        WHERE loan_status IN (
            'Charged Off', 'Default',
            'Does not meet the credit policy. Status:Charged Off'
        )
        AND funded_amnt > 0
        GROUP BY grade
        ORDER BY grade
    """


def _q_lgd_by_term():
    """LGD statistics by term."""
    return """
        SELECT
            term,
            COUNT(*)::numeric                                    AS total_defaulted,
            ROUND(AVG(1 - (total_pymnt / funded_amnt))::numeric, 4) AS avg_lgd,
            ROUND(AVG(total_pymnt / funded_amnt)::numeric, 4)   AS avg_recovery_rate,
            ROUND(AVG(funded_amnt)::numeric, 2)                 AS avg_exposure,
            ROUND(AVG(funded_amnt - total_pymnt)::numeric, 2)   AS avg_loss_amount
        FROM loans_master
        WHERE loan_status IN (
            'Charged Off', 'Default',
            'Does not meet the credit policy. Status:Charged Off'
        )
        AND funded_amnt > 0
        GROUP BY term
        ORDER BY term
    """


def _q_el_by_grade():
    """Expected loss breakdown by grade: EL = PD x LGD x EAD."""
    return """
        WITH mature AS (
            SELECT * FROM loans_master
            WHERE loan_status IN (
                'Fully Paid', 'Charged Off', 'Default',
                'Does not meet the credit policy. Status:Fully Paid',
                'Does not meet the credit policy. Status:Charged Off'
            )
        ),
        pd_by_grade AS (
            SELECT grade,
                ROUND(SUM(CASE WHEN loan_status IN (
                    'Charged Off', 'Default',
                    'Does not meet the credit policy. Status:Charged Off'
                ) THEN 1 ELSE 0 END)::numeric / COUNT(*), 4) AS pd
            FROM mature GROUP BY grade
        ),
        lgd_by_grade AS (
            SELECT grade,
                ROUND(AVG(1 - (total_pymnt / funded_amnt))::numeric, 4) AS lgd
            FROM loans_master
            WHERE loan_status IN ('Charged Off', 'Default',
                'Does not meet the credit policy. Status:Charged Off')
            AND funded_amnt > 0
            GROUP BY grade
        )
        SELECT
            m.grade,
            COUNT(*)::numeric                                    AS loan_count,
            ROUND(p.pd * 100, 2)                                AS pd_pct,
            ROUND(l.lgd * 100, 2)                               AS lgd_pct,
            ROUND(AVG(m.funded_amnt)::numeric, 2)               AS avg_ead,
            ROUND(SUM(m.funded_amnt)::numeric, 2)               AS total_ead,
            ROUND(p.pd * l.lgd * AVG(m.funded_amnt)::numeric, 2) AS el_per_loan,
            ROUND(p.pd * l.lgd * SUM(m.funded_amnt)::numeric, 2) AS total_el,
            ROUND(p.pd * l.lgd * 100, 4)                        AS el_rate_pct
        FROM mature m
        JOIN pd_by_grade p ON m.grade = p.grade
        JOIN lgd_by_grade l ON m.grade = l.grade
        GROUP BY m.grade, p.pd, l.lgd
        ORDER BY m.grade
    """


def _q_concentration_by_grade():
    """Portfolio concentration by grade (HHI contribution)."""
    return """
        WITH totals AS (
            SELECT COUNT(*)::numeric AS n, SUM(loan_amnt)::numeric AS total_amt
            FROM loans_master
        )
        SELECT
            'grade'::text AS segment_type,
            grade AS segment,
            COUNT(*)::numeric                                    AS count,
            ROUND(SUM(loan_amnt)::numeric, 2)                   AS total_amount,
            ROUND(100.0 * SUM(loan_amnt)::numeric / (SELECT total_amt FROM totals), 2) AS pct,
            ROUND(
                POWER(100.0 * SUM(loan_amnt)::numeric / (SELECT total_amt FROM totals), 2)::numeric, 2
            ) AS hhi_contrib
        FROM loans_master, totals
        GROUP BY grade, totals.total_amt
        ORDER BY total_amount DESC
    """


def _q_concentration_by_purpose():
    """Portfolio concentration by loan purpose."""
    return """
        WITH totals AS (
            SELECT SUM(loan_amnt)::numeric AS total_amt FROM loans_master
        )
        SELECT
            'purpose'::text AS segment_type,
            purpose AS segment,
            COUNT(*)::numeric                                    AS count,
            ROUND(SUM(loan_amnt)::numeric, 2)                   AS total_amount,
            ROUND(100.0 * SUM(loan_amnt)::numeric / (SELECT total_amt FROM totals), 2) AS pct,
            ROUND(
                POWER(100.0 * SUM(loan_amnt)::numeric / (SELECT total_amt FROM totals), 2)::numeric, 2
            ) AS hhi_contrib
        FROM loans_master, totals
        GROUP BY purpose, totals.total_amt
        ORDER BY total_amount DESC
    """


def _q_concentration_by_state():
    """Portfolio concentration by state (from zip_code prefix)."""
    return """
        WITH totals AS (
            SELECT SUM(loan_amnt)::numeric AS total_amt FROM loans_master
        )
        SELECT
            'state'::text AS segment_type,
            SUBSTRING(zip_code, 1, 1) AS segment,
            COUNT(*)::numeric                                    AS count,
            ROUND(SUM(loan_amnt)::numeric, 2)                   AS total_amount,
            ROUND(100.0 * SUM(loan_amnt)::numeric / (SELECT total_amt FROM totals), 2) AS pct,
            ROUND(
                POWER(100.0 * SUM(loan_amnt)::numeric / (SELECT total_amt FROM totals), 2)::numeric, 2
            ) AS hhi_contrib
        FROM loans_master, totals
        GROUP BY SUBSTRING(zip_code, 1, 1), totals.total_amt
        ORDER BY total_amount DESC
        """


def _q_concentration_by_term():
    """Portfolio concentration by term."""
    return """
        WITH totals AS (
            SELECT SUM(loan_amnt)::numeric AS total_amt FROM loans_master
        )
        SELECT
            'term'::text AS segment_type,
            term AS segment,
            COUNT(*)::numeric                                    AS count,
            ROUND(SUM(loan_amnt)::numeric, 2)                   AS total_amount,
            ROUND(100.0 * SUM(loan_amnt)::numeric / (SELECT total_amt FROM totals), 2) AS pct,
            ROUND(
                POWER(100.0 * SUM(loan_amnt)::numeric / (SELECT total_amt FROM totals), 2)::numeric, 2
            ) AS hhi_contrib
        FROM loans_master, totals
        GROUP BY term, totals.total_amt
        ORDER BY total_amount DESC
    """


def _q_int_rate_by_grade():
    """Interest rate statistics by grade."""
    return """
        SELECT
            grade,
            COUNT(*)::numeric                                    AS count,
            ROUND(MIN(int_rate)::numeric, 2)                    AS min_rate,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (
                ORDER BY int_rate)::numeric, 2)                  AS p25_rate,
            ROUND(AVG(int_rate)::numeric, 2)                    AS avg_rate,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (
                ORDER BY int_rate)::numeric, 2)                  AS p75_rate,
            ROUND(MAX(int_rate)::numeric, 2)                    AS max_rate,
            ROUND(STDDEV(int_rate)::numeric, 2)                 AS std_rate
        FROM loans_master
        GROUP BY grade
        ORDER BY grade
    """


def _q_fico_by_grade():
    """FICO score distribution by grade."""
    return """
        SELECT
            grade,
            COUNT(*)::numeric                                    AS count,
            ROUND(AVG((fico_range_low + fico_range_high) / 2.0)::numeric, 1) AS avg_fico,
            ROUND(MIN(fico_range_low)::numeric, 0)              AS min_fico_low,
            ROUND(MAX(fico_range_high)::numeric, 0)             AS max_fico_high,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (
                ORDER BY (fico_range_low + fico_range_high) / 2.0)::numeric, 1) AS p25_fico,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (
                ORDER BY (fico_range_low + fico_range_high) / 2.0)::numeric, 1) AS median_fico,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (
                ORDER BY (fico_range_low + fico_range_high) / 2.0)::numeric, 1) AS p75_fico
        FROM loans_master
        GROUP BY grade
        ORDER BY grade
    """


def _q_dti_by_grade():
    """Debt-to-income ratio distribution by grade."""
    return """
        SELECT
            grade,
            COUNT(*)::numeric                                    AS count,
            ROUND(AVG(dti)::numeric, 2)                         AS avg_dti,
            ROUND(MIN(dti)::numeric, 2)                         AS min_dti,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (
                ORDER BY dti)::numeric, 2)                       AS p25_dti,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (
                ORDER BY dti)::numeric, 2)                       AS median_dti,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (
                ORDER BY dti)::numeric, 2)                       AS p75_dti,
            ROUND(MAX(dti)::numeric, 2)                         AS max_dti
        FROM loans_master
        GROUP BY grade
        ORDER BY grade
    """


def _q_issuance_by_month():
    """Monthly loan issuance trend."""
    return """
        SELECT
            TO_CHAR(issue_d, 'YYYY-MM') AS month,
            COUNT(*)::numeric                                    AS loans_issued,
            ROUND(SUM(loan_amnt)::numeric, 2)                   AS total_amount,
            ROUND(AVG(loan_amnt)::numeric, 2)                   AS avg_amount,
            ROUND(AVG(int_rate)::numeric, 2)                    AS avg_int_rate
        FROM loans_master
        WHERE issue_d IS NOT NULL
        GROUP BY TO_CHAR(issue_d, 'YYYY-MM')
        ORDER BY month
    """


def _q_issuance_by_year():
    """Yearly loan issuance trend."""
    return """
        SELECT
            EXTRACT(YEAR FROM issue_d)::int AS year,
            COUNT(*)::numeric                                    AS loans_issued,
            ROUND(SUM(loan_amnt)::numeric, 2)                   AS total_amount,
            ROUND(AVG(loan_amnt)::numeric, 2)                   AS avg_amount,
            ROUND(AVG(int_rate)::numeric, 2)                    AS avg_int_rate
        FROM loans_master
        WHERE issue_d IS NOT NULL
        GROUP BY EXTRACT(YEAR FROM issue_d)
        ORDER BY year
    """


def _q_annual_income_by_grade():
    """Annual income statistics by grade."""
    return """
        SELECT
            grade,
            COUNT(*)::numeric                                    AS count,
            ROUND(AVG(annual_inc)::numeric, 2)                  AS avg_income,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (
                ORDER BY annual_inc)::numeric, 2)                AS p25_income,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (
                ORDER BY annual_inc)::numeric, 2)                AS median_income,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (
                ORDER BY annual_inc)::numeric, 2)                AS p75_income
        FROM loans_master
        GROUP BY grade
        ORDER BY grade
    """


def _q_verification_by_grade():
    """Verification status distribution by grade."""
    return """
        SELECT
            grade,
            verification_status,
            COUNT(*)::numeric                                    AS count,
            ROUND(100.0 * COUNT(*)::numeric / SUM(COUNT(*)) OVER (
                PARTITION BY grade), 2)                          AS pct_within_grade
        FROM loans_master
        GROUP BY grade, verification_status
        ORDER BY grade, count DESC
    """


def _q_zip_summary():
    """Summary statistics by 3-digit ZIP code (top 50 by loan count)."""
    return """
        SELECT
            zip_code,
            COUNT(*)::numeric                                    AS total_loans,
            ROUND(SUM(loan_amnt)::numeric, 2)                   AS total_amount,
            ROUND(AVG(loan_amnt)::numeric, 2)                   AS avg_amount,
            ROUND(AVG(int_rate)::numeric, 2)                    AS avg_int_rate,
            ROUND(AVG((fico_range_low + fico_range_high) / 2.0)::numeric, 1) AS avg_fico,
            ROUND(AVG(annual_inc)::numeric, 2)                  AS avg_income,
            ROUND(AVG(median_income_2024)::numeric, 2)          AS avg_census_income,
            ROUND(AVG(income_growth_23_24)::numeric, 4)         AS avg_income_growth
        FROM loans_master
        WHERE zip_code IS NOT NULL
        GROUP BY zip_code
        ORDER BY total_loans DESC
        LIMIT 500
    """


def _q_delinquency_by_grade():
    """Delinquency (2-year) analysis by grade."""
    return """
        SELECT
            grade,
            COUNT(*)::numeric                                    AS total_loans,
            SUM(CASE WHEN delinq_2yrs > 0 THEN 1 ELSE 0 END)::numeric AS loans_with_delinq,
            ROUND(100.0 * SUM(CASE WHEN delinq_2yrs > 0 THEN 1 ELSE 0 END)::numeric
                / COUNT(*), 2)                                  AS delinq_rate_pct,
            ROUND(AVG(delinq_2yrs)::numeric, 2)                 AS avg_delinq_count
        FROM loans_master
        GROUP BY grade
        ORDER BY grade
    """


def _q_inquiry_by_grade():
    """Inquiry (6-month) analysis by grade."""
    return """
        SELECT
            grade,
            COUNT(*)::numeric                                    AS total_loans,
            ROUND(AVG(inq_last_6mths)::numeric, 2)             AS avg_inquiries,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (
                ORDER BY inq_last_6mths)::numeric, 1)           AS p75_inquiries,
            SUM(CASE WHEN inq_last_6mths >= 3 THEN 1 ELSE 0 END)::numeric AS high_inquiry_count,
            ROUND(100.0 * SUM(CASE WHEN inq_last_6mths >= 3 THEN 1 ELSE 0 END)::numeric
                / COUNT(*), 2)                                  AS high_inquiry_pct
        FROM loans_master
        GROUP BY grade
        ORDER BY grade
    """


def _q_census_enrichment():
    """Census economic indicators by ZIP code region."""
    return """
        SELECT
            zip_code,
            COUNT(*)::numeric                                    AS loan_count,
            ROUND(AVG(median_income_2024)::numeric, 2)          AS avg_median_income,
            ROUND(AVG(housing_cost_2024)::numeric, 2)           AS avg_housing_cost,
            ROUND(AVG(income_growth_22_23)::numeric, 4)         AS avg_growth_22_23,
            ROUND(AVG(income_growth_23_24)::numeric, 4)         AS avg_growth_23_24,
            ROUND(AVG(int_rate)::numeric, 2)                    AS avg_int_rate,
            ROUND(AVG((fico_range_low + fico_range_high) / 2.0)::numeric, 1) AS avg_fico
        FROM loans_master
        WHERE zip_code IS NOT NULL AND median_income_2024 > 0
        GROUP BY zip_code
        HAVING COUNT(*) >= 100
        ORDER BY avg_median_income DESC
    """


# ==================================================================
# Table Registry
# ==================================================================

TABLE_REGISTRY = [
    # (filename, description, query_func)
    ("portfolio_by_grade",         "Portfolio summary by grade",                _q_portfolio_by_grade),
    ("portfolio_by_term",          "Portfolio summary by term",                 _q_portfolio_by_term),
    ("portfolio_by_grade_term",    "Portfolio cross-tab grade x term",         _q_portfolio_by_grade_term),
    ("loan_status",                "Loan status distribution",                 _q_loan_status),
    ("default_by_grade",           "Default rate by grade (mature)",           _q_default_rates_by_grade),
    ("default_by_term",            "Default rate by term",                     _q_default_rates_by_term),
    ("default_by_purpose",         "Default rate by loan purpose",             _q_default_rates_by_purpose),
    ("default_by_home_ownership",  "Default rate by home ownership",           _q_default_rates_by_home_ownership),
    ("lgd_by_grade",               "LGD statistics by grade",                  _q_lgd_by_grade),
    ("lgd_by_term",                "LGD statistics by term",                   _q_lgd_by_term),
    ("expected_loss",              "Expected loss by grade",                   _q_el_by_grade),
    ("concentration_grade",        "Concentration HHI by grade",               _q_concentration_by_grade),
    ("concentration_purpose",      "Concentration HHI by purpose",             _q_concentration_by_purpose),
    ("concentration_state",        "Concentration HHI by ZIP region",          _q_concentration_by_state),
    ("concentration_term",         "Concentration HHI by term",                _q_concentration_by_term),
    ("int_rate_by_grade",          "Interest rate stats by grade",             _q_int_rate_by_grade),
    ("fico_by_grade",              "FICO score distribution by grade",         _q_fico_by_grade),
    ("dti_by_grade",               "DTI distribution by grade",                _q_dti_by_grade),
    ("issuance_by_month",          "Monthly issuance trend",                   _q_issuance_by_month),
    ("issuance_by_year",           "Yearly issuance trend",                    _q_issuance_by_year),
    ("income_by_grade",            "Annual income by grade",                   _q_annual_income_by_grade),
    ("verification_by_grade",      "Verification status by grade",             _q_verification_by_grade),
    ("zip_summary",                "ZIP code summary (top 500)",               _q_zip_summary),
    ("delinquency_by_grade",       "Delinquency analysis by grade",            _q_delinquency_by_grade),
    ("inquiry_by_grade",           "Inquiry analysis by grade",                _q_inquiry_by_grade),
    ("census_enrichment",          "Census enrichment by ZIP",                 _q_census_enrichment),
]


# ==================================================================
# Export Logic
# ==================================================================

def _export_csv(df, filename):
    """Save a DataFrame to CSV and return the file path."""
    filepath = _OUTPUT_DIR / "{}.csv".format(filename)
    df.to_csv(filepath, index=False)
    return filepath


def export_powerbi_data():
    """
    Run all 27 SQL queries and export results as CSV to data/powerbi/.

    Also copies model dashboard JSON files into the output directory
    so Power BI can consume them directly.

    Returns
    -------
    dict
        Mapping of filename -> row count for each exported table.
    """
    engine = get_engine()
    if not engine:
        raise ConnectionError("Database connection failed.")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("POWER BI DATA EXPORT")
    print("  Output: {}".format(_OUTPUT_DIR))
    print("  Tables: {}".format(len(TABLE_REGISTRY)))
    print("=" * 60)

    results = {}
    total_rows = 0

    for idx, (filename, description, query_func) in enumerate(TABLE_REGISTRY, 1):
        tag = "{:02d}/{}".format(idx, len(TABLE_REGISTRY))
        try:
            sql = query_func()
            df = pd.read_sql(sql, engine)
            filepath = _export_csv(df, filename)
            results[filename] = len(df)
            total_rows += len(df)
            print("   [{}] {:<30} {:>6,} rows  {}".format(
                tag, filename, len(df), description
            ))
        except Exception as exc:
            print("   [{}] {:<30} FAILED: {}".format(tag, filename, exc))
            results[filename] = 0

    # ----------------------------------------------------------
    # Copy model dashboard JSONs if available
    # ----------------------------------------------------------
    model_dir = config.BASE_DIR / "output" / "models"
    json_files = {
        "pd_model_metrics": "pd_model_metrics.json",
        "lgd_model_metrics": "lgd_model_metrics.json",
        "vasicek_results": "vasicek_results.json",
    }
    for dest_name, src_file in json_files.items():
        src_path = model_dir / src_file
        if src_path.exists():
            import shutil
            dest_path = _OUTPUT_DIR / src_file
            shutil.copy2(src_path, dest_path)
            print("   Copied model artifact: {}".format(src_file))

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("-" * 60)
    print("   Total: {} tables, {:,} rows exported".format(
        len([v for v in results.values() if v > 0]), total_rows
    ))
    print("   Output directory: {}".format(_OUTPUT_DIR))
    print("=" * 60)

    return results


# ==================================================================
# Standalone Execution
# ==================================================================

if __name__ == "__main__":
    try:
        results = export_powerbi_data()
        print("\nDone. {} tables exported.".format(len(results)))
    except Exception as exc:
        print("ERROR: {}".format(exc))
        import traceback
        traceback.print_exc()
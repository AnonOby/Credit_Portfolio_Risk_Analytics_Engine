"""
Data Fetcher - Shared database query utility for the visualization module.

All heavy SQL queries are centralised here so every page can reuse them.
Results are cached via @st.cache_data to avoid hammering PostgreSQL.
"""

import sys
from pathlib import Path

import pandas as pd

# Make sure project root is on sys.path so we can import sibling packages
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.database import get_engine


# ---------------------------------------------------------------------------
# Portfolio overview queries
# ---------------------------------------------------------------------------

def get_portfolio_summary() -> pd.DataFrame:
    """Return grade-grouped portfolio overview: count, total funded amount, avg interest rate, avg term."""
    sql = """
    SELECT
        grade,
        COUNT(*)                                            AS total_loans,
        SUM(funded_amnt)                                    AS total_funded,
        ROUND(AVG(funded_amnt)::numeric, 2)                 AS avg_loan_amount,
        ROUND(AVG(int_rate)::numeric, 2)                    AS avg_int_rate,
        ROUND(AVG(annual_inc)::numeric, 2)                  AS avg_annual_income,
        ROUND(AVG(dti)::numeric, 2)                         AS avg_dti,
        ROUND(AVG(fico_range_low)::numeric, 2)              AS avg_fico_low,
        ROUND(AVG(fico_range_high)::numeric, 2)             AS avg_fico_high
    FROM loans_master
    GROUP BY grade
    ORDER BY grade;
    """
    return pd.read_sql(sql, get_engine())


def get_loan_status_distribution() -> pd.DataFrame:
    """Return count and percentage per loan_status."""
    sql = """
    SELECT
        loan_status,
        COUNT(*)                                        AS count,
        ROUND(COUNT(*)::numeric * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
    FROM loans_master
    GROUP BY loan_status
    ORDER BY count DESC;
    """
    return pd.read_sql(sql, get_engine())


def get_funded_amount_distribution() -> pd.DataFrame:
    """Return funded amount histogram buckets (25 bins)."""
    sql = """
    SELECT
        bucket,
        COUNT(*)                                        AS count,
        (bucket - 1) * 2000                             AS low,
        bucket * 2000                                   AS high
    FROM (
        SELECT width_bucket(funded_amnt, 0, 40000, 20) AS bucket
        FROM loans_master
    ) t
    GROUP BY bucket
    ORDER BY bucket;
    """
    return pd.read_sql(sql, get_engine())


def get_grade_distribution() -> pd.DataFrame:
    """Return loan count per grade."""
    sql = """
    SELECT grade, COUNT(*) AS count
    FROM loans_master
    GROUP BY grade
    ORDER BY grade;
    """
    return pd.read_sql(sql, get_engine())


def get_term_distribution() -> pd.DataFrame:
    """Return loan count per term."""
    sql = """
    SELECT
        term,
        COUNT(*) AS count
    FROM loans_master
    GROUP BY term
    ORDER BY term;
    """
    return pd.read_sql(sql, get_engine())


def get_purpose_distribution() -> pd.DataFrame:
    """Return loan count and total funded per purpose."""
    sql = """
    SELECT
        purpose,
        COUNT(*)                                        AS count,
        SUM(funded_amnt)                                AS total_funded
    FROM loans_master
    GROUP BY purpose
    ORDER BY count DESC
    LIMIT 15;
    """
    return pd.read_sql(sql, get_engine())


def get_issuance_trend() -> pd.DataFrame:
    """Return monthly loan issuance counts."""
    sql = """
    SELECT
        DATE_TRUNC('month', issue_d::date)             AS month,
        COUNT(*)                                        AS count,
        SUM(funded_amnt)                                AS total_funded
    FROM loans_master
    WHERE issue_d IS NOT NULL
    GROUP BY month
    ORDER BY month;
    """
    return pd.read_sql(sql, get_engine())


def get_state_distribution() -> pd.DataFrame:
    """Return loan count per state (top 20)."""
    sql = """
    SELECT
        addr_state,
        COUNT(*)                                        AS count,
        SUM(funded_amnt)                                AS total_funded
    FROM loans_master
    WHERE addr_state IS NOT NULL
    GROUP BY addr_state
    ORDER BY count DESC
    LIMIT 20;
    """
    return pd.read_sql(sql, get_engine())


def get_home_ownership_distribution() -> pd.DataFrame:
    """Return loan count per home ownership type."""
    sql = """
    SELECT
        home_ownership,
        COUNT(*)                                        AS count,
        ROUND(AVG(funded_amnt)::numeric, 2)             AS avg_funded,
        ROUND(AVG(int_rate)::numeric, 2)                AS avg_int_rate
    FROM loans_master
    GROUP BY home_ownership
    ORDER BY count DESC;
    """
    return pd.read_sql(sql, get_engine())


# ---------------------------------------------------------------------------
# Default analysis queries
# ---------------------------------------------------------------------------

def get_default_rates_by_grade() -> pd.DataFrame:
    """Return default rates per grade (mature loans only)."""
    sql = """
    WITH mature AS (
        SELECT *,
            CASE WHEN loan_status IN (
                'Charged Off', 'Default',
                'Does not meet the credit policy. Status:Charged Off'
            ) THEN TRUE ELSE FALSE END AS is_default
        FROM loans_master
        WHERE loan_status IN (
            'Fully Paid', 'Charged Off', 'Default',
            'Does not meet the credit policy. Status:Fully Paid',
            'Does not meet the credit policy. Status:Charged Off'
        )
    )
    SELECT
        grade,
        COUNT(*)                                        AS total_mature,
        SUM(CASE WHEN is_default THEN 1 ELSE 0 END)    AS defaults,
        ROUND(
            SUM(CASE WHEN is_default THEN 1 ELSE 0 END)::numeric
            * 100.0 / COUNT(*), 2
        )                                               AS default_rate
    FROM mature
    GROUP BY grade
    ORDER BY grade;
    """
    return pd.read_sql(sql, get_engine())


def get_default_rates_over_time() -> pd.DataFrame:
    """Return monthly default rate trend (mature loans)."""
    sql = """
    WITH mature AS (
        SELECT *,
            DATE_TRUNC('month', issue_d::date)          AS issue_month,
            CASE WHEN loan_status IN (
                'Charged Off', 'Default',
                'Does not meet the credit policy. Status:Charged Off'
            ) THEN TRUE ELSE FALSE END AS is_default
        FROM loans_master
        WHERE loan_status IN (
            'Fully Paid', 'Charged Off', 'Default',
            'Does not meet the credit policy. Status:Fully Paid',
            'Does not meet the credit policy. Status:Charged Off'
        )
        AND issue_d IS NOT NULL
    )
    SELECT
        issue_month,
        COUNT(*)                                        AS total,
        SUM(CASE WHEN is_default THEN 1 ELSE 0 END)    AS defaults,
        ROUND(
            SUM(CASE WHEN is_default THEN 1 ELSE 0 END)::numeric
            * 100.0 / COUNT(*), 2
        )                                               AS default_rate
    FROM mature
    GROUP BY issue_month
    HAVING COUNT(*) >= 100
    ORDER BY issue_month;
    """
    return pd.read_sql(sql, get_engine())


def get_default_by_purpose() -> pd.DataFrame:
    """Return default rates per loan purpose (mature loans)."""
    sql = """
    WITH mature AS (
        SELECT *,
            CASE WHEN loan_status IN (
                'Charged Off', 'Default',
                'Does not meet the credit policy. Status:Charged Off'
            ) THEN TRUE ELSE FALSE END AS is_default
        FROM loans_master
        WHERE loan_status IN (
            'Fully Paid', 'Charged Off', 'Default',
            'Does not meet the credit policy. Status:Fully Paid',
            'Does not meet the credit policy. Status:Charged Off'
        )
    )
    SELECT
        purpose,
        COUNT(*)                                        AS total,
        SUM(CASE WHEN is_default THEN 1 ELSE 0 END)    AS defaults,
        ROUND(
            SUM(CASE WHEN is_default THEN 1 ELSE 0 END)::numeric
            * 100.0 / COUNT(*), 2
        )                                               AS default_rate
    FROM mature
    GROUP BY purpose
    HAVING COUNT(*) >= 100
    ORDER BY default_rate DESC;
    """
    return pd.read_sql(sql, get_engine())


def get_default_by_home_ownership() -> pd.DataFrame:
    """Return default rates per home ownership type (mature loans)."""
    sql = """
    WITH mature AS (
        SELECT *,
            CASE WHEN loan_status IN (
                'Charged Off', 'Default',
                'Does not meet the credit policy. Status:Charged Off'
            ) THEN TRUE ELSE FALSE END AS is_default
        FROM loans_master
        WHERE loan_status IN (
            'Fully Paid', 'Charged Off', 'Default',
            'Does not meet the credit policy. Status:Fully Paid',
            'Does not meet the credit policy. Status:Charged Off'
        )
    )
    SELECT
        home_ownership,
        COUNT(*)                                        AS total,
        SUM(CASE WHEN is_default THEN 1 ELSE 0 END)    AS defaults,
        ROUND(
            SUM(CASE WHEN is_default THEN 1 ELSE 0 END)::numeric
            * 100.0 / COUNT(*), 2
        )                                               AS default_rate
    FROM mature
    GROUP BY home_ownership
    ORDER BY default_rate DESC;
    """
    return pd.read_sql(sql, get_engine())


def get_default_by_dti_bucket() -> pd.DataFrame:
    """Return default rates by DTI buckets (mature loans)."""
    sql = """
    WITH mature AS (
        SELECT *,
            CASE WHEN loan_status IN (
                'Charged Off', 'Default',
                'Does not meet the credit policy. Status:Charged Off'
            ) THEN TRUE ELSE FALSE END AS is_default
        FROM loans_master
        WHERE loan_status IN (
            'Fully Paid', 'Charged Off', 'Default',
            'Does not meet the credit policy. Status:Fully Paid',
            'Does not meet the credit policy. Status:Charged Off'
        )
        AND dti IS NOT NULL
    ),
    bucketed AS (
        SELECT
            WIDTH_BUCKET(dti, 0, 50, 10) AS bucket,
            is_default
        FROM mature
    )
    SELECT
        bucket,
        (bucket - 1) * 5 AS dti_low,
        bucket * 5 AS dti_high,
        COUNT(*) AS total,
        SUM(CASE WHEN is_default THEN 1 ELSE 0 END) AS defaults,
        ROUND(
            100.0 * SUM(CASE WHEN is_default THEN 1 ELSE 0 END) / COUNT(*), 2
        ) AS default_rate
    FROM bucketed
    GROUP BY bucket
    ORDER BY bucket;
    """
    return pd.read_sql(sql, get_engine())


def get_default_by_fico_bucket() -> pd.DataFrame:
    """Return default rates by FICO score buckets (mature loans)."""
    sql = """
    WITH mature AS (
        SELECT *,
            (fico_range_low + fico_range_high) / 2.0    AS fico_midpoint,
            CASE WHEN loan_status IN (
                'Charged Off', 'Default',
                'Does not meet the credit policy. Status:Charged Off'
            ) THEN TRUE ELSE FALSE END AS is_default
        FROM loans_master
        WHERE loan_status IN (
            'Fully Paid', 'Charged Off', 'Default',
            'Does not meet the credit policy. Status:Fully Paid',
            'Does not meet the credit policy. Status:Charged Off'
        )
        AND fico_range_low IS NOT NULL
    )
    SELECT
        WIDTH_BUCKET(fico_midpoint, 640, 850, 10)      AS bucket,
        (bucket - 1) * 21 + 640                         AS fico_low,
        bucket * 21 + 640                               AS fico_high,
        COUNT(*)                                        AS total,
        SUM(CASE WHEN is_default THEN 1 ELSE 0 END)    AS defaults,
        ROUND(
            SUM(CASE WHEN is_default THEN 1 ELSE 0 END)::numeric
            * 100.0 / COUNT(*), 2
        )                                               AS default_rate
    FROM mature
    GROUP BY bucket
    ORDER BY bucket;
    """
    return pd.read_sql(sql, get_engine())


# ---------------------------------------------------------------------------
# Risk metrics queries
# ---------------------------------------------------------------------------

def get_lgd_by_grade() -> pd.DataFrame:
    """Return LGD statistics per grade."""
    sql = """
    SELECT
        grade,
        COUNT(*)                                        AS total_defaulted,
        ROUND(AVG(1.0 - total_pymnt / funded_amnt)::numeric, 4) AS avg_lgd,
        ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY 1.0 - total_pymnt / funded_amnt)::numeric, 4) AS p25_lgd,
        ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY 1.0 - total_pymnt / funded_amnt)::numeric, 4) AS p50_lgd,
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY 1.0 - total_pymnt / funded_amnt)::numeric, 4) AS p75_lgd
    FROM loans_master
    WHERE loan_status IN (
        'Charged Off', 'Default',
        'Does not meet the credit policy. Status:Charged Off'
    )
    AND funded_amnt > 0
    GROUP BY grade
    ORDER BY grade;
    """
    return pd.read_sql(sql, get_engine())


def get_el_by_grade() -> pd.DataFrame:
    """Return expected loss by grade: PD * LGD * avg EAD."""
    sql = """
    WITH mature AS (
        SELECT *,
            CASE WHEN loan_status IN (
                'Charged Off', 'Default',
                'Does not meet the credit policy. Status:Charged Off'
            ) THEN 1 ELSE 0 END AS is_default
        FROM loans_master
        WHERE loan_status IN (
            'Fully Paid', 'Charged Off', 'Default',
            'Does not meet the credit policy. Status:Fully Paid',
            'Does not meet the credit policy. Status:Charged Off'
        )
    ),
    pd_by_grade AS (
        SELECT
            grade,
            ROUND(AVG(is_default)::numeric, 4)         AS pd
        FROM mature
        GROUP BY grade
    ),
    lgd_by_grade AS (
        SELECT
            grade,
            ROUND(AVG(1.0 - total_pymnt / funded_amnt)::numeric, 4) AS lgd
        FROM loans_master
        WHERE loan_status IN (
            'Charged Off', 'Default',
            'Does not meet the credit policy. Status:Charged Off'
        )
        AND funded_amnt > 0
        GROUP BY grade
    ),
    seg AS (
        SELECT
            grade,
            COUNT(*)                                    AS n,
            SUM(funded_amnt)                            AS total_ead,
            ROUND(AVG(funded_amnt)::numeric, 2)         AS avg_ead
        FROM loans_master
        GROUP BY grade
    )
    SELECT
        seg.grade,
        seg.n,
        seg.total_ead,
        seg.avg_ead,
        pd_by_grade.pd,
        lgd_by_grade.lgd,
        ROUND((pd_by_grade.pd * lgd_by_grade.lgd * seg.avg_ead)::numeric, 2) AS el_per_loan,
        ROUND((pd_by_grade.pd * lgd_by_grade.lgd * seg.total_ead)::numeric, 2) AS total_el
    FROM seg
    JOIN pd_by_grade ON seg.grade = pd_by_grade.grade
    JOIN lgd_by_grade ON seg.grade = lgd_by_grade.grade
    ORDER BY seg.grade;
    """
    return pd.read_sql(sql, get_engine())


def get_concentration_metrics() -> pd.DataFrame:
    """Return HHI concentration index by grade, purpose, and state."""
    sql = """
    -- HHI by grade
    SELECT 'grade' AS segment_type, grade AS segment,
        ROUND((COUNT(*)::numeric / SUM(COUNT(*)) OVER () * 100)::numeric, 2) AS pct,
        ROUND(POWER(COUNT(*)::numeric / SUM(COUNT(*)) OVER (), 2) * 10000::numeric, 2) AS hhi_contrib
    FROM loans_master
    GROUP BY grade
    UNION ALL
    -- HHI by purpose (top 10)
    SELECT 'purpose' AS segment_type, purpose AS segment,
        ROUND((COUNT(*)::numeric / SUM(COUNT(*)) OVER () * 100)::numeric, 2) AS pct,
        ROUND(POWER(COUNT(*)::numeric / SUM(COUNT(*)) OVER (), 2) * 10000::numeric, 2) AS hhi_contrib
    FROM loans_master
    GROUP BY purpose
    ORDER BY segment_type, hhi_contrib DESC;
    """
    return pd.read_sql(sql, get_engine())


def get_int_rate_distribution_by_grade() -> pd.DataFrame:
    """Return avg/min/max interest rate per grade."""
    sql = """
    SELECT
        grade,
        COUNT(*)                                        AS count,
        ROUND(MIN(int_rate)::numeric, 2)                AS min_rate,
        ROUND(AVG(int_rate)::numeric, 2)                AS avg_rate,
        ROUND(MAX(int_rate)::numeric, 2)                AS max_rate
    FROM loans_master
    WHERE int_rate IS NOT NULL
    GROUP BY grade
    ORDER BY grade;
    """
    return pd.read_sql(sql, get_engine())


def get_emp_length_distribution() -> pd.DataFrame:
    """Return loan count per employment length."""
    sql = """
    SELECT
        emp_length,
        COUNT(*)                                        AS count,
        ROUND(AVG(funded_amnt)::numeric, 2)             AS avg_funded,
        ROUND(AVG(int_rate)::numeric, 2)                AS avg_int_rate
    FROM loans_master
    WHERE emp_length IS NOT NULL
    GROUP BY emp_length
    ORDER BY count DESC;
    """
    return pd.read_sql(sql, get_engine())


def get_annual_income_distribution() -> pd.DataFrame:
    """Return income statistics and loan count per income bucket."""
    sql = """
    SELECT
        WIDTH_BUCKET(annual_inc, 0, 300000, 15)        AS bucket,
        (bucket - 1) * 20000                            AS income_low,
        bucket * 20000                                  AS income_high,
        COUNT(*)                                        AS count,
        ROUND(AVG(funded_amnt)::numeric, 2)             AS avg_funded,
        ROUND(AVG(int_rate)::numeric, 2)                AS avg_int_rate,
        ROUND(AVG(dti)::numeric, 2)                     AS avg_dti
    FROM loans_master
    WHERE annual_inc IS NOT NULL AND annual_inc < 300000
    GROUP BY bucket
    ORDER BY bucket;
    """
    return pd.read_sql(sql, get_engine())


# ---------------------------------------------------------------------------
# Convenience wrapper class
# ---------------------------------------------------------------------------

class DataFetcher:
    """
    Centralised data access for the Streamlit dashboard.
    Each method returns a DataFrame ready for Plotly / Streamlit rendering.
    """

    @staticmethod
    def portfolio_summary():
        return get_portfolio_summary()

    @staticmethod
    def loan_status_distribution():
        return get_loan_status_distribution()

    @staticmethod
    def grade_distribution():
        return get_grade_distribution()

    @staticmethod
    def term_distribution():
        return get_term_distribution()

    @staticmethod
    def purpose_distribution():
        return get_purpose_distribution()

    @staticmethod
    def issuance_trend():
        return get_issuance_trend()

    @staticmethod
    def state_distribution():
        return get_state_distribution()

    @staticmethod
    def home_ownership_distribution():
        return get_home_ownership_distribution()

    @staticmethod
    def funded_amount_distribution():
        return get_funded_amount_distribution()

    @staticmethod
    def default_rates_by_grade():
        return get_default_rates_by_grade()

    @staticmethod
    def default_rates_over_time():
        return get_default_rates_over_time()

    @staticmethod
    def default_by_purpose():
        return get_default_by_purpose()

    @staticmethod
    def default_by_home_ownership():
        return get_default_by_home_ownership()

    @staticmethod
    def default_by_dti_bucket():
        return get_default_by_dti_bucket()

    @staticmethod
    def default_by_fico_bucket():
        return get_default_by_fico_bucket()

    @staticmethod
    def lgd_by_grade():
        return get_lgd_by_grade()

    @staticmethod
    def el_by_grade():
        return get_el_by_grade()

    @staticmethod
    def concentration_metrics():
        return get_concentration_metrics()

    @staticmethod
    def int_rate_by_grade():
        return get_int_rate_distribution_by_grade()

    @staticmethod
    def emp_length_distribution():
        return get_emp_length_distribution()

    @staticmethod
    def income_distribution():
        return get_annual_income_distribution()
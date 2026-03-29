-- =============================================================
-- 05_risk_summary.sql
-- Executive Risk Summary — Run directly in pgAdmin / DBeaver
-- Combines PD, LGD, and EL into a single comprehensive view
-- with macroeconomic enrichment from Census data
-- =============================================================

-- -----------------------------------------------
-- 1. Executive Risk Summary by Grade
-- Combines PD, LGD, EAD, and EL into one table
-- -----------------------------------------------
WITH mature_loans AS (
    SELECT *
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
        COUNT(*) AS total_loans,
        SUM(CASE WHEN loan_status IN ('Charged Off', 'Default',
            'Does not meet the credit policy. Status:Charged Off') THEN 1 ELSE 0 END) AS defaulted,
        ROUND(
            SUM(CASE WHEN loan_status IN ('Charged Off', 'Default',
                'Does not meet the credit policy. Status:Charged Off') THEN 1 ELSE 0 END) * 1.0
            / COUNT(*)::numeric, 4) AS pd
    FROM mature_loans
    GROUP BY grade
),
lgd_by_grade AS (
    SELECT
        grade,
        COUNT(*) AS defaulted_count,
        ROUND(AVG(1 - (total_pymnt / funded_amnt))::numeric, 4) AS lgd,
        ROUND(AVG(total_pymnt / funded_amnt)::numeric, 4) AS recovery_rate
    FROM loans_master
    WHERE loan_status IN ('Charged Off', 'Default',
        'Does not meet the credit policy. Status:Charged Off')
    AND funded_amnt > 0
    GROUP BY grade
),
el_calc AS (
    SELECT
        p.grade,
        p.total_loans,
        p.defaulted,
        ROUND((p.pd * 100)::numeric, 2)              AS pd_pct,
        ROUND((l.lgd * 100)::numeric, 2)              AS lgd_pct,
        ROUND((l.recovery_rate * 100)::numeric, 2)    AS recovery_pct,
        ROUND(AVG(m.funded_amnt)::numeric, 2)         AS avg_ead,
        ROUND(SUM(m.funded_amnt)::numeric, 2)         AS total_exposure,
        ROUND((p.pd * l.lgd * AVG(m.funded_amnt))::numeric, 2) AS el_per_loan,
        ROUND((p.pd * l.lgd * SUM(m.funded_amnt))::numeric, 2) AS total_el
    FROM pd_by_grade p
    JOIN lgd_by_grade l ON p.grade = l.grade
    JOIN mature_loans m ON p.grade = m.grade
    GROUP BY p.grade, p.total_loans, p.defaulted, p.pd, l.lgd, l.recovery_rate
)
SELECT
    grade,
    total_loans,
    defaulted,
    pd_pct,
    lgd_pct,
    recovery_pct,
    avg_ead,
    total_exposure,
    el_per_loan,
    total_el
FROM el_calc
ORDER BY grade;


-- -----------------------------------------------
-- 2. Risk-Adjusted Metrics by Grade Group
-- Tiered view: Prime / Near-Prime / Subprime
-- -----------------------------------------------
WITH mature_loans AS (
    SELECT *,
        CASE
            WHEN grade IN ('A', 'B') THEN '1_Prime (A-B)'
            WHEN grade IN ('C', 'D') THEN '2_Near-Prime (C-D)'
            WHEN grade IN ('E', 'F', 'G') THEN '3_Subprime (E-G)'
        END AS risk_tier
    FROM loans_master
    WHERE loan_status IN (
        'Fully Paid', 'Charged Off', 'Default',
        'Does not meet the credit policy. Status:Fully Paid',
        'Does not meet the credit policy. Status:Charged Off'
    )
),
pd_by_tier AS (
    SELECT
        risk_tier,
        ROUND(
            SUM(CASE WHEN loan_status IN ('Charged Off', 'Default',
                'Does not meet the credit policy. Status:Charged Off') THEN 1 ELSE 0 END) * 1.0
            / COUNT(*)::numeric, 4) AS pd
    FROM mature_loans
    GROUP BY risk_tier
),
lgd_by_tier AS (
    SELECT
        CASE
            WHEN grade IN ('A', 'B') THEN '1_Prime (A-B)'
            WHEN grade IN ('C', 'D') THEN '2_Near-Prime (C-D)'
            WHEN grade IN ('E', 'F', 'G') THEN '3_Subprime (E-G)'
        END AS risk_tier,
        ROUND(AVG(1 - (total_pymnt / funded_amnt))::numeric, 4) AS lgd
    FROM loans_master
    WHERE loan_status IN ('Charged Off', 'Default',
        'Does not meet the credit policy. Status:Charged Off')
    AND funded_amnt > 0
    GROUP BY risk_tier
),
el_calc AS (
    SELECT
        m.risk_tier,
        COUNT(*) AS loan_count,
        ROUND(SUM(m.funded_amnt)::numeric, 2)      AS total_exposure,
        p.pd,
        l.lgd
    FROM mature_loans m
    JOIN pd_by_tier p ON m.risk_tier = p.risk_tier
    JOIN lgd_by_tier l ON m.risk_tier = l.risk_tier
    GROUP BY m.risk_tier, p.pd, l.lgd
)
SELECT
    risk_tier,
    loan_count,
    total_exposure,
    ROUND((pd * 100)::numeric, 2)                  AS pd_pct,
    ROUND((lgd * 100)::numeric, 2)                 AS lgd_pct,
    ROUND((pd * lgd * 100)::numeric, 4)            AS el_rate_pct,
    ROUND((pd * lgd * total_exposure)::numeric, 2) AS total_expected_loss
FROM el_calc
ORDER BY risk_tier;


-- -----------------------------------------------
-- 3. Geographic Risk: Top 20 ZIP Regions by EL Rate
-- Uses Census-enriched income growth data
-- -----------------------------------------------
WITH mature_loans AS (
    SELECT *
    FROM loans_master
    WHERE loan_status IN (
        'Fully Paid', 'Charged Off', 'Default',
        'Does not meet the credit policy. Status:Fully Paid',
        'Does not meet the credit policy. Status:Charged Off'
    )
),
pd_by_zip AS (
    SELECT
        zip_code,
        ROUND(
            SUM(CASE WHEN loan_status IN ('Charged Off', 'Default',
                'Does not meet the credit policy. Status:Charged Off') THEN 1 ELSE 0 END) * 1.0
            / NULLIF(COUNT(*), 0)::numeric, 4) AS pd,
        COUNT(*) AS loan_count
    FROM mature_loans
    WHERE zip_code IS NOT NULL
    GROUP BY zip_code
    HAVING COUNT(*) >= 500
),
lgd_by_zip AS (
    SELECT
        zip_code,
        ROUND(AVG(1 - (total_pymnt / funded_amnt))::numeric, 4) AS lgd
    FROM loans_master
    WHERE loan_status IN ('Charged Off', 'Default',
        'Does not meet the credit policy. Status:Charged Off')
    AND funded_amnt > 0
    AND zip_code IS NOT NULL
    GROUP BY zip_code
    HAVING COUNT(*) >= 100
)
SELECT
    p.zip_code,
    p.loan_count,
    ROUND((p.pd * 100)::numeric, 2)                  AS pd_pct,
    ROUND((l.lgd * 100)::numeric, 2)                 AS lgd_pct,
    ROUND((p.pd * l.lgd * 100)::numeric, 4)          AS el_rate_pct,
    ROUND(AVG(m.median_income_2024)::numeric, 2)     AS avg_median_income,
    ROUND(AVG(m.income_growth_22_23)::numeric, 4)    AS avg_income_growth_22_23,
    ROUND(AVG(m.income_growth_23_24)::numeric, 4)    AS avg_income_growth_23_24
FROM pd_by_zip p
JOIN lgd_by_zip l ON p.zip_code = l.zip_code
LEFT JOIN loans_master m ON p.zip_code = m.zip_code
GROUP BY p.zip_code, p.loan_count, p.pd, l.lgd
ORDER BY el_rate_pct DESC
LIMIT 20;


-- -----------------------------------------------
-- 4. Macroeconomic Risk Factor Analysis
-- Correlation between Census income growth and default rates
-- -----------------------------------------------
WITH mature_loans AS (
    SELECT *
    FROM loans_master
    WHERE loan_status IN (
        'Fully Paid', 'Charged Off', 'Default',
        'Does not meet the credit policy. Status:Fully Paid',
        'Does not meet the credit policy. Status:Charged Off'
    )
    AND zip_code IS NOT NULL
),
zip_risk AS (
    SELECT
        zip_code,
        ROUND(
            SUM(CASE WHEN loan_status IN ('Charged Off', 'Default',
                'Does not meet the credit policy. Status:Charged Off') THEN 1 ELSE 0 END) * 100.0
            / NULLIF(COUNT(*), 0)::numeric, 2) AS default_rate_pct,
        COUNT(*) AS loan_count,
        ROUND(AVG(median_income_2024)::numeric, 2)       AS avg_median_income,
        ROUND(AVG(income_growth_22_23)::numeric, 4)      AS avg_income_growth_22_23,
        ROUND(AVG(income_growth_23_24)::numeric, 4)      AS avg_income_growth_23_24
    FROM mature_loans
    GROUP BY zip_code
    HAVING COUNT(*) >= 500
)
SELECT
    CASE
        WHEN avg_income_growth_23_24 < -0.05 THEN 'Strong Decline (< -5%)'
        WHEN avg_income_growth_23_24 < -0.02 THEN 'Moderate Decline (-5% to -2%)'
        WHEN avg_income_growth_23_24 < 0.0    THEN 'Slight Decline (-2% to 0%)'
        WHEN avg_income_growth_23_24 < 0.02   THEN 'Slight Growth (0% to 2%)'
        WHEN avg_income_growth_23_24 < 0.05   THEN 'Moderate Growth (2% to 5%)'
        ELSE 'Strong Growth (> 5%)'
    END AS income_growth_bucket,
    COUNT(*) AS num_zip_regions,
    SUM(loan_count) AS total_loans,
    ROUND(AVG(default_rate_pct)::numeric, 2)  AS avg_default_rate_pct,
    ROUND(AVG(avg_median_income)::numeric, 2) AS avg_median_income
FROM zip_risk
WHERE avg_income_growth_23_24 IS NOT NULL
GROUP BY income_growth_bucket
ORDER BY avg_default_rate_pct DESC;
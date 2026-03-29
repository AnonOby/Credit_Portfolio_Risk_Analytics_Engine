-- =============================================================
-- 02_default_rate_analysis.sql
-- Default Rate Analysis — Run directly in pgAdmin / DBeaver
-- =============================================================

-- -----------------------------------------------
-- 1. Default Rate by Grade
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
labeled AS (
    SELECT *,
        CASE WHEN loan_status IN ('Charged Off', 'Default',
            'Does not meet the credit policy. Status:Charged Off') THEN 1 ELSE 0
        END AS is_default
    FROM mature_loans
)
SELECT
    grade,
    COUNT(*)                                        AS total_loans,
    SUM(is_default)                                 AS defaulted,
    ROUND(100.0 * SUM(is_default) / COUNT(*), 2)   AS default_rate_pct,
    ROUND(AVG(loan_amnt), 2)                        AS avg_loan_amount,
    ROUND(AVG(int_rate), 2)                         AS avg_int_rate,
    ROUND(AVG((fico_range_low + fico_range_high) / 2.0), 1) AS avg_fico,
    ROUND(AVG(annual_inc), 2)                       AS avg_annual_income,
    ROUND(AVG(dti), 2)                              AS avg_dti
FROM labeled
GROUP BY grade
ORDER BY grade;


-- -----------------------------------------------
-- 2. Default Rate by Term
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
labeled AS (
    SELECT *,
        CASE WHEN loan_status IN ('Charged Off', 'Default',
            'Does not meet the credit policy. Status:Charged Off') THEN 1 ELSE 0
        END AS is_default
    FROM mature_loans
)
SELECT
    term,
    COUNT(*)                                        AS total_loans,
    SUM(is_default)                                 AS defaulted,
    ROUND(100.0 * SUM(is_default) / COUNT(*), 2)   AS default_rate_pct
FROM labeled
GROUP BY term
ORDER BY term;


-- -----------------------------------------------
-- 3. Default Rate by Purpose (Top 15)
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
labeled AS (
    SELECT *,
        CASE WHEN loan_status IN ('Charged Off', 'Default',
            'Does not meet the credit policy. Status:Charged Off') THEN 1 ELSE 0
        END AS is_default
    FROM mature_loans
)
SELECT
    purpose,
    COUNT(*)                                        AS total_loans,
    SUM(is_default)                                 AS defaulted,
    ROUND(100.0 * SUM(is_default) / COUNT(*), 2)   AS default_rate_pct,
    ROUND(AVG(int_rate), 2)                         AS avg_int_rate
FROM labeled
GROUP BY purpose
HAVING COUNT(*) >= 1000
ORDER BY default_rate_pct DESC
LIMIT 15;


-- -----------------------------------------------
-- 4. Default Rate by Home Ownership
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
labeled AS (
    SELECT *,
        CASE WHEN loan_status IN ('Charged Off', 'Default',
            'Does not meet the credit policy. Status:Charged Off') THEN 1 ELSE 0
        END AS is_default
    FROM mature_loans
)
SELECT
    home_ownership,
    COUNT(*)                                        AS total_loans,
    SUM(is_default)                                 AS defaulted,
    ROUND(100.0 * SUM(is_default) / COUNT(*), 2)   AS default_rate_pct
FROM labeled
GROUP BY home_ownership
ORDER BY default_rate_pct DESC;


-- -----------------------------------------------
-- 5. Default Rate by FICO Range (Binned)
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
labeled AS (
    SELECT *,
        CASE WHEN loan_status IN ('Charged Off', 'Default',
            'Does not meet the credit policy. Status:Charged Off') THEN 1 ELSE 0
        END AS is_default,
        WIDTH_BUCKET((fico_range_low + fico_range_high) / 2.0, 640, 850, 7) AS fico_bucket
    FROM mature_loans
),
fico_labels AS (
    SELECT *,
        CASE fico_bucket
            WHEN 1 THEN '640-660'
            WHEN 2 THEN '660-680'
            WHEN 3 THEN '680-700'
            WHEN 4 THEN '700-720'
            WHEN 5 THEN '720-740'
            WHEN 6 THEN '740-760'
            WHEN 7 THEN '760-780'
            WHEN 8 THEN '780+'
            ELSE 'Below 640'
        END AS fico_range_label
    FROM labeled
)
SELECT
    fico_range_label,
    COUNT(*)                                        AS total_loans,
    SUM(is_default)                                 AS defaulted,
    ROUND(100.0 * SUM(is_default) / COUNT(*), 2)   AS default_rate_pct
FROM fico_labels
WHERE fico_range_label IS NOT NULL
GROUP BY fico_range_label
ORDER BY fico_bucket;


-- -----------------------------------------------
-- 6. Default Rate by Verification Status
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
labeled AS (
    SELECT *,
        CASE WHEN loan_status IN ('Charged Off', 'Default',
            'Does not meet the credit policy. Status:Charged Off') THEN 1 ELSE 0
        END AS is_default
    FROM mature_loans
)
SELECT
    verification_status,
    COUNT(*)                                        AS total_loans,
    SUM(is_default)                                 AS defaulted,
    ROUND(100.0 * SUM(is_default) / COUNT(*), 2)   AS default_rate_pct
FROM labeled
GROUP BY verification_status
ORDER BY default_rate_pct DESC;


-- -----------------------------------------------
-- 7. Default Rate by Issue Year (Time Trend)
-- -----------------------------------------------
WITH mature_loans AS (
    SELECT *,
        CASE WHEN loan_status IN ('Charged Off', 'Default',
            'Does not meet the credit policy. Status:Charged Off') THEN 1 ELSE 0
        END AS is_default
    FROM loans_master
    WHERE loan_status IN (
        'Fully Paid', 'Charged Off', 'Default',
        'Does not meet the credit policy. Status:Fully Paid',
        'Does not meet the credit policy. Status:Charged Off'
    )
)
SELECT
    EXTRACT(YEAR FROM issue_d)::int                 AS issue_year,
    COUNT(*)                                        AS total_loans,
    SUM(is_default)                                 AS defaulted,
    ROUND(100.0 * SUM(is_default) / COUNT(*), 2)   AS default_rate_pct,
    ROUND(AVG(loan_amnt), 2)                        AS avg_loan_amount
FROM mature_loans
GROUP BY EXTRACT(YEAR FROM issue_d)
ORDER BY issue_year;
-- =============================================================
-- 04_el_by_grade.sql
-- Expected Loss by Grade — Run directly in pgAdmin / DBeaver
-- EL = PD x EAD x LGD
-- =============================================================

-- -----------------------------------------------
-- 1. Full EL Breakdown by Grade
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
        ROUND(
            SUM(CASE WHEN loan_status IN ('Charged Off', 'Default',
                'Does not meet the credit policy. Status:Charged Off') THEN 1 ELSE 0 END) * 1.0
            / COUNT(*), 4) AS pd
    FROM mature_loans
    GROUP BY grade
),
lgd_by_grade AS (
    SELECT
        grade,
        ROUND(AVG(1 - (total_pymnt / funded_amnt)), 4) AS lgd
    FROM loans_master
    WHERE loan_status IN ('Charged Off', 'Default',
        'Does not meet the credit policy. Status:Charged Off')
    AND funded_amnt > 0
    GROUP BY grade
),
el_calc AS (
    SELECT
        m.grade,
        COUNT(*)                                    AS loan_count,
        ROUND(AVG(m.funded_amnt), 2)               AS avg_ead,
        ROUND(SUM(m.funded_amnt), 2)               AS total_exposure,
        p.pd,
        l.lgd
    FROM mature_loans m
    JOIN pd_by_grade p ON m.grade = p.grade
    JOIN lgd_by_grade l ON m.grade = l.grade
    GROUP BY m.grade, p.pd, l.lgd
)
SELECT
    grade,
    loan_count,
    total_exposure,
    ROUND(pd * 100, 2)                             AS pd_pct,
    ROUND(lgd * 100, 2)                            AS lgd_pct,
    avg_ead,
    ROUND(pd * lgd * avg_ead, 2)                    AS expected_loss_per_loan,
    ROUND(pd * lgd * total_exposure, 2)             AS total_expected_loss,
    ROUND(pd * lgd * 100, 4)                        AS el_rate_pct
FROM el_calc
ORDER BY grade;


-- -----------------------------------------------
-- 2. EL by Grade Group (A-B / C-D / E-G)
-- -----------------------------------------------
WITH mature_loans AS (
    SELECT *,
        CASE
            WHEN grade IN ('A', 'B') THEN 'A-B (Prime)'
            WHEN grade IN ('C', 'D') THEN 'C-D (Near-Prime)'
            WHEN grade IN ('E', 'F', 'G') THEN 'E-G (Subprime)'
        END AS grade_group
    FROM loans_master
    WHERE loan_status IN (
        'Fully Paid', 'Charged Off', 'Default',
        'Does not meet the credit policy. Status:Fully Paid',
        'Does not meet the credit policy. Status:Charged Off'
    )
),
pd_by_group AS (
    SELECT
        grade_group,
        ROUND(
            SUM(CASE WHEN loan_status IN ('Charged Off', 'Default',
                'Does not meet the credit policy. Status:Charged Off') THEN 1 ELSE 0 END) * 1.0
            / COUNT(*), 4) AS pd
    FROM mature_loans
    GROUP BY grade_group
),
lgd_by_group AS (
    SELECT
        CASE
            WHEN grade IN ('A', 'B') THEN 'A-B (Prime)'
            WHEN grade IN ('C', 'D') THEN 'C-D (Near-Prime)'
            WHEN grade IN ('E', 'F', 'G') THEN 'E-G (Subprime)'
        END AS grade_group,
        ROUND(AVG(1 - (total_pymnt / funded_amnt)), 4) AS lgd
    FROM loans_master
    WHERE loan_status IN ('Charged Off', 'Default',
        'Does not meet the credit policy. Status:Charged Off')
    AND funded_amnt > 0
    GROUP BY grade_group
),
el_calc AS (
    SELECT
        m.grade_group,
        COUNT(*)                                    AS loan_count,
        ROUND(SUM(m.funded_amnt), 2)               AS total_exposure,
        p.pd,
        l.lgd
    FROM mature_loans m
    JOIN pd_by_group p ON m.grade_group = p.grade_group
    JOIN lgd_by_group l ON m.grade_group = l.grade_group
    GROUP BY m.grade_group, p.pd, l.lgd
)
SELECT
    grade_group,
    loan_count,
    total_exposure,
    ROUND(pd * 100, 2)                             AS pd_pct,
    ROUND(lgd * 100, 2)                            AS lgd_pct,
    ROUND(pd * lgd * 100, 4)                        AS el_rate_pct,
    ROUND(pd * lgd * total_exposure, 2)             AS total_expected_loss
FROM el_calc
ORDER BY grade_group;


-- -----------------------------------------------
-- 3. EL by Term
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
pd_by_term AS (
    SELECT
        term,
        ROUND(
            SUM(CASE WHEN loan_status IN ('Charged Off', 'Default',
                'Does not meet the credit policy. Status:Charged Off') THEN 1 ELSE 0 END) * 1.0
            / COUNT(*), 4) AS pd
    FROM mature_loans
    GROUP BY term
),
lgd_by_term AS (
    SELECT
        term,
        ROUND(AVG(1 - (total_pymnt / funded_amnt)), 4) AS lgd
    FROM loans_master
    WHERE loan_status IN ('Charged Off', 'Default',
        'Does not meet the credit policy. Status:Charged Off')
    AND funded_amnt > 0
    GROUP BY term
),
el_calc AS (
    SELECT
        m.term,
        COUNT(*)                                    AS loan_count,
        ROUND(SUM(m.funded_amnt), 2)               AS total_exposure,
        p.pd,
        l.lgd
    FROM mature_loans m
    JOIN pd_by_term p ON m.term = p.term
    JOIN lgd_by_term l ON m.term = l.term
    GROUP BY m.term, p.pd, l.lgd
)
SELECT
    term,
    loan_count,
    total_exposure,
    ROUND(pd * 100, 2)                             AS pd_pct,
    ROUND(lgd * 100, 2)                            AS lgd_pct,
    ROUND(pd * lgd * 100, 4)                        AS el_rate_pct,
    ROUND(pd * lgd * total_exposure, 2)             AS total_expected_loss
FROM el_calc
ORDER BY term;


-- -----------------------------------------------
-- 4. EL by Purpose (Top 10 by total EL)
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
pd_by_purpose AS (
    SELECT
        purpose,
        ROUND(
            SUM(CASE WHEN loan_status IN ('Charged Off', 'Default',
                'Does not meet the credit policy. Status:Charged Off') THEN 1 ELSE 0 END) * 1.0
            / COUNT(*), 4) AS pd
    FROM mature_loans
    GROUP BY purpose
),
lgd_by_purpose AS (
    SELECT
        purpose,
        ROUND(AVG(1 - (total_pymnt / funded_amnt)), 4) AS lgd
    FROM loans_master
    WHERE loan_status IN ('Charged Off', 'Default',
        'Does not meet the credit policy. Status:Charged Off')
    AND funded_amnt > 0
    GROUP BY purpose
),
el_calc AS (
    SELECT
        m.purpose,
        COUNT(*)                                    AS loan_count,
        ROUND(SUM(m.funded_amnt), 2)               AS total_exposure,
        p.pd,
        l.lgd
    FROM mature_loans m
    JOIN pd_by_purpose p ON m.purpose = p.purpose
    JOIN lgd_by_purpose l ON m.purpose = l.purpose
    GROUP BY m.purpose, p.pd, l.lgd
)
SELECT
    purpose,
    loan_count,
    total_exposure,
    ROUND(pd * 100, 2)                             AS pd_pct,
    ROUND(lgd * 100, 2)                            AS lgd_pct,
    ROUND(pd * lgd * 100, 4)                        AS el_rate_pct,
    ROUND(pd * lgd * total_exposure, 2)             AS total_expected_loss
FROM el_calc
ORDER BY total_expected_loss DESC
LIMIT 10;
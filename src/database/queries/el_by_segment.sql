-- =============================================================
-- Expected Loss (EL) by Segment
-- EL = PD x EAD x LGD
-- PD  : Historical default rate per grade (mature loans only)
-- EAD : Average funded amount per grade (mature loans only)
-- LGD : Average loss given default per grade (defaulted loans only)
-- =============================================================

WITH mature_loans AS (
    SELECT *
    FROM loans_master
    WHERE loan_status IN (
        'Fully Paid',
        'Charged Off',
        'Default',
        'Does not meet the credit policy. Status:Fully Paid',
        'Does not meet the credit policy. Status:Charged Off'
    )
),
pd_by_grade AS (
    SELECT
        grade,
        ROUND(
            SUM(CASE
                WHEN loan_status IN ('Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off')
                THEN 1 ELSE 0
            END) * 1.0 / COUNT(*),
        4) AS pd
    FROM mature_loans
    GROUP BY grade
),
lgd_by_grade AS (
    SELECT
        grade,
        ROUND(AVG(1 - (total_pymnt / funded_amnt)), 4) AS lgd
    FROM loans_master
    WHERE loan_status IN ('Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off')
    AND funded_amnt > 0
    GROUP BY grade
),
segment_metrics AS (
    SELECT
        m.grade,
        COUNT(*)                                    AS loan_count,
        ROUND(AVG(m.funded_amnt), 2)               AS avg_ead,
        ROUND(SUM(m.funded_amnt), 2)               AS total_exposure,
        p.pd,
        l.lgd,
        ROUND(p.pd * l.lgd, 4)                     AS el_rate
    FROM mature_loans m
    JOIN pd_by_grade p ON m.grade = p.grade
    JOIN lgd_by_grade l ON m.grade = l.grade
    GROUP BY m.grade, p.pd, l.lgd
)
SELECT
    grade,
    loan_count,
    ROUND(pd * 100, 2)                             AS pd_pct,
    ROUND(lgd * 100, 2)                            AS lgd_pct,
    avg_ead,
    total_exposure,
    ROUND(el_rate * avg_ead, 2)                    AS el_per_loan,
    ROUND(el_rate * total_exposure, 2)             AS total_el,
    ROUND(el_rate * 100, 4)                        AS portfolio_loss_rate_pct
FROM segment_metrics
ORDER BY grade;
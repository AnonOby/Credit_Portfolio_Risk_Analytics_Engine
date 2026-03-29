-- =============================================================
-- Default Rate Analysis by Grade
-- Default = 'Charged Off', 'Default', or credit policy rejection variants
-- Only considers mature loans (fully paid or charged off / defaulted)
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
labeled AS (
    SELECT *,
        CASE
            WHEN loan_status IN (
                'Charged Off',
                'Default',
                'Does not meet the credit policy. Status:Charged Off'
            ) THEN 1
            ELSE 0
        END AS is_default
    FROM mature_loans
)
SELECT
    grade,
    COUNT(*)                                            AS total_mature_loans,
    SUM(is_default)                                     AS defaulted_loans,
    ROUND(100.0 * SUM(is_default) / COUNT(*)::numeric, 2) AS default_rate_pct,
    ROUND(AVG(loan_amnt)::numeric, 2)                   AS avg_loan_amount,
    ROUND(AVG(int_rate)::numeric, 2)                    AS avg_interest_rate,
    ROUND(AVG((fico_range_low + fico_range_high) / 2.0)::numeric, 1) AS avg_fico_score,
    ROUND(AVG(annual_inc)::numeric, 2)                  AS avg_annual_income,
    ROUND(AVG(dti)::numeric, 2)                         AS avg_dti,
    ROUND(AVG(emp_length)::numeric, 1)                  AS avg_emp_length_years,
    ROUND(AVG(credit_history_months)::numeric, 1)       AS avg_credit_history_months
FROM labeled
GROUP BY grade
ORDER BY grade;
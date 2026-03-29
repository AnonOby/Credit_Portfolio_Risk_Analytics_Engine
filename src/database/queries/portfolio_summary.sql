-- =============================================================
-- Portfolio Summary
-- Provides a high-level overview of the loan portfolio
-- =============================================================

SELECT
    grade,
    COUNT(*)                                            AS total_loans,
    ROUND(SUM(loan_amnt), 2)                            AS total_loan_amount,
    ROUND(AVG(loan_amnt), 2)                            AS avg_loan_amount,
    ROUND(AVG(funded_amnt), 2)                          AS avg_funded_amount,
    ROUND(AVG(int_rate), 2)                             AS avg_interest_rate,
    ROUND(AVG((fico_range_low + fico_range_high) / 2.0), 1) AS avg_fico_score,
    ROUND(AVG(annual_inc), 2)                           AS avg_annual_income,
    ROUND(AVG(dti), 2)                                  AS avg_dti,
    ROUND(AVG(emp_length), 1)                           AS avg_emp_length_years,
    ROUND(AVG(credit_history_months), 1)                AS avg_credit_history_months
FROM loans_master
GROUP BY grade
ORDER BY grade;
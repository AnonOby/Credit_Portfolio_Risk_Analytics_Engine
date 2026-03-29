-- =============================================================
-- 01_portfolio_overview.sql
-- Portfolio Overview — Run directly in pgAdmin / DBeaver
-- =============================================================

-- -----------------------------------------------
-- 1. Overall Portfolio Snapshot
-- -----------------------------------------------
SELECT
    COUNT(*)                                                AS total_loans,
    COUNT(DISTINCT grade)                                   AS num_grades,
    COUNT(DISTINCT purpose)                                 AS num_purposes,
    ROUND(SUM(loan_amnt), 2)                                AS total_loan_amount,
    ROUND(AVG(loan_amnt), 2)                                AS avg_loan_amount,
    ROUND(SUM(funded_amnt), 2)                               AS total_funded_amount,
    ROUND(AVG(funded_amnt), 2)                               AS avg_funded_amount,
    ROUND(AVG(int_rate), 2)                                 AS avg_interest_rate,
    ROUND(MIN(int_rate), 2)                                 AS min_interest_rate,
    ROUND(MAX(int_rate), 2)                                 AS max_interest_rate,
    ROUND(AVG(annual_inc), 2)                               AS avg_annual_income,
    ROUND(AVG(dti), 2)                                      AS avg_dti,
    ROUND(AVG((fico_range_low + fico_range_high) / 2.0), 1) AS avg_fico_score,
    MIN(issue_d)                                            AS earliest_issue_date,
    MAX(issue_d)                                            AS latest_issue_date,
    COUNT(DISTINCT zip_code)                                AS num_zip_regions
FROM loans_master;


-- -----------------------------------------------
-- 2. Loan Distribution by Grade
-- -----------------------------------------------
SELECT
    grade,
    COUNT(*)                                AS total_loans,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_portfolio,
    ROUND(SUM(loan_amnt), 2)                AS total_loan_amount,
    ROUND(AVG(loan_amnt), 2)                AS avg_loan_amount,
    ROUND(AVG(int_rate), 2)                 AS avg_interest_rate,
    ROUND(AVG((fico_range_low + fico_range_high) / 2.0), 1) AS avg_fico
FROM loans_master
GROUP BY grade
ORDER BY grade;


-- -----------------------------------------------
-- 3. Loan Distribution by Purpose (Top 15)
-- -----------------------------------------------
SELECT
    purpose,
    COUNT(*)                                AS total_loans,
    ROUND(AVG(loan_amnt), 2)                AS avg_loan_amount,
    ROUND(AVG(int_rate), 2)                 AS avg_interest_rate,
    ROUND(AVG((fico_range_low + fico_range_high) / 2.0), 1) AS avg_fico,
    ROUND(AVG(annual_inc), 2)               AS avg_annual_income
FROM loans_master
GROUP BY purpose
ORDER BY total_loans DESC
LIMIT 15;


-- -----------------------------------------------
-- 4. Loan Distribution by Term
-- -----------------------------------------------
SELECT
    term,
    COUNT(*)                                AS total_loans,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_portfolio,
    ROUND(AVG(loan_amnt), 2)                AS avg_loan_amount,
    ROUND(AVG(int_rate), 2)                 AS avg_interest_rate
FROM loans_master
GROUP BY term
ORDER BY term;


-- -----------------------------------------------
-- 5. Loan Distribution by Home Ownership
-- -----------------------------------------------
SELECT
    home_ownership,
    COUNT(*)                                AS total_loans,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_portfolio,
    ROUND(AVG(loan_amnt), 2)                AS avg_loan_amount,
    ROUND(AVG(int_rate), 2)                 AS avg_interest_rate,
    ROUND(AVG(annual_inc), 2)               AS avg_annual_income
FROM loans_master
GROUP BY home_ownership
ORDER BY total_loans DESC;


-- -----------------------------------------------
-- 6. Loan Status Distribution
-- -----------------------------------------------
SELECT
    loan_status,
    COUNT(*)                                AS total_loans,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_portfolio,
    ROUND(AVG(loan_amnt), 2)                AS avg_loan_amount
FROM loans_master
GROUP BY loan_status
ORDER BY total_loans DESC;


-- -----------------------------------------------
-- 7. FICO Score Distribution by Grade
-- -----------------------------------------------
SELECT
    grade,
    ROUND(AVG(fico_range_low), 1)           AS avg_fico_low,
    ROUND(AVG(fico_range_high), 1)          AS avg_fico_high,
    ROUND(AVG((fico_range_low + fico_range_high) / 2.0), 1) AS avg_fico_midpoint,
    ROUND(MIN(fico_range_low), 1)           AS min_fico_low,
    ROUND(MAX(fico_range_high), 1)          AS max_fico_high
FROM loans_master
GROUP BY grade
ORDER BY grade;
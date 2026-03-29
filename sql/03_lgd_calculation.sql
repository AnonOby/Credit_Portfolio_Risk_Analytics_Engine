-- =============================================================
-- 03_lgd_calculation.sql
-- LGD Calculation — Run directly in pgAdmin / DBeaver
-- LGD = 1 - (total_pymnt / funded_amnt)
-- =============================================================

-- -----------------------------------------------
-- 1. Overall LGD Statistics
-- -----------------------------------------------
SELECT
    COUNT(*)                                                        AS total_defaulted_loans,
    ROUND(AVG(1 - (total_pymnt / funded_amnt)), 4)                 AS overall_avg_lgd,
    ROUND(AVG(total_pymnt / funded_amnt), 4)                       AS overall_avg_recovery_rate,
    ROUND(AVG(funded_amnt), 2)                                     AS avg_exposure_at_default,
    ROUND(AVG(funded_amnt - total_pymnt), 2)                       AS avg_loss_amount,
    ROUND(SUM(funded_amnt - total_pymnt), 2)                       AS total_loss_amount,
    ROUND(
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY 1 - total_pymnt / funded_amnt), 4
    )                                                               AS median_lgd,
    ROUND(
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY 1 - total_pymnt / funded_amnt), 4
    )                                                               AS p25_lgd,
    ROUND(
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY 1 - total_pymnt / funded_amnt), 4
    )                                                               AS p75_lgd
FROM loans_master
WHERE loan_status IN ('Charged Off', 'Default',
    'Does not meet the credit policy. Status:Charged Off')
AND funded_amnt > 0;


-- -----------------------------------------------
-- 2. LGD by Grade
-- -----------------------------------------------
SELECT
    grade,
    COUNT(*)                                                        AS total_defaulted,
    ROUND(AVG(1 - (total_pymnt / funded_amnt)), 4)                 AS avg_lgd,
    ROUND(AVG(total_pymnt / funded_amnt), 4)                       AS avg_recovery_rate,
    ROUND(AVG(funded_amnt), 2)                                     AS avg_exposure_at_default,
    ROUND(AVG(funded_amnt - total_pymnt), 2)                       AS avg_loss_amount,
    ROUND(SUM(funded_amnt - total_pymnt), 2)                       AS total_loss_amount,
    ROUND(
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY 1 - total_pymnt / funded_amnt), 4
    )                                                               AS median_lgd,
    ROUND(
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY 1 - total_pymnt / funded_amnt), 4
    )                                                               AS p25_lgd,
    ROUND(
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY 1 - total_pymnt / funded_amnt), 4
    )                                                               AS p75_lgd
FROM loans_master
WHERE loan_status IN ('Charged Off', 'Default',
    'Does not meet the credit policy. Status:Charged Off')
AND funded_amnt > 0
GROUP BY grade
ORDER BY grade;


-- -----------------------------------------------
-- 3. LGD by Grade Group (A-B / C-D / E-G)
-- -----------------------------------------------
SELECT
    CASE
        WHEN grade IN ('A', 'B') THEN 'A-B (Prime)'
        WHEN grade IN ('C', 'D') THEN 'C-D (Near-Prime)'
        WHEN grade IN ('E', 'F', 'G') THEN 'E-G (Subprime)'
    END AS grade_group,
    COUNT(*)                                                        AS total_defaulted,
    ROUND(AVG(1 - (total_pymnt / funded_amnt)), 4)                 AS avg_lgd,
    ROUND(AVG(total_pymnt / funded_amnt), 4)                       AS avg_recovery_rate,
    ROUND(AVG(funded_amnt - total_pymnt), 2)                       AS avg_loss_amount,
    ROUND(SUM(funded_amnt - total_pymnt), 2)                       AS total_loss_amount
FROM loans_master
WHERE loan_status IN ('Charged Off', 'Default',
    'Does not meet the credit policy. Status:Charged Off')
AND funded_amnt > 0
GROUP BY grade_group
ORDER BY grade_group;


-- -----------------------------------------------
-- 4. LGD by Term
-- -----------------------------------------------
SELECT
    term,
    COUNT(*)                                                        AS total_defaulted,
    ROUND(AVG(1 - (total_pymnt / funded_amnt)), 4)                 AS avg_lgd,
    ROUND(AVG(total_pymnt / funded_amnt), 4)                       AS avg_recovery_rate,
    ROUND(AVG(funded_amnt), 2)                                     AS avg_exposure_at_default
FROM loans_master
WHERE loan_status IN ('Charged Off', 'Default',
    'Does not meet the credit policy. Status:Charged Off')
AND funded_amnt > 0
GROUP BY term
ORDER BY term;


-- -----------------------------------------------
-- 5. LGD by Purpose (Top 10 by volume)
-- -----------------------------------------------
SELECT
    purpose,
    COUNT(*)                                                        AS total_defaulted,
    ROUND(AVG(1 - (total_pymnt / funded_amnt)), 4)                 AS avg_lgd,
    ROUND(AVG(total_pymnt / funded_amnt), 4)                       AS avg_recovery_rate,
    ROUND(AVG(funded_amnt), 2)                                     AS avg_exposure_at_default
FROM loans_master
WHERE loan_status IN ('Charged Off', 'Default',
    'Does not meet the credit policy. Status:Charged Off')
AND funded_amnt > 0
GROUP BY purpose
HAVING COUNT(*) >= 500
ORDER BY total_defaulted DESC
LIMIT 10;


-- -----------------------------------------------
-- 6. Recovery Rate Distribution (Binned)
-- -----------------------------------------------
SELECT
    CASE
        WHEN total_pymnt / funded_amnt >= 1.0 THEN '100%+ (Full Recovery)'
        WHEN total_pymnt / funded_amnt >= 0.8 THEN '80-100%'
        WHEN total_pymnt / funded_amnt >= 0.6 THEN '60-80%'
        WHEN total_pymnt / funded_amnt >= 0.4 THEN '40-60%'
        WHEN total_pymnt / funded_amnt >= 0.2 THEN '20-40%'
        WHEN total_pymnt / funded_amnt >= 0.0 THEN '0-20%'
        ELSE 'Negative Recovery'
    END AS recovery_bucket,
    COUNT(*)                                                        AS total_loans,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)             AS pct_of_defaulted
FROM loans_master
WHERE loan_status IN ('Charged Off', 'Default',
    'Does not meet the credit policy. Status:Charged Off')
AND funded_amnt > 0
GROUP BY recovery_bucket
ORDER BY recovery_bucket;
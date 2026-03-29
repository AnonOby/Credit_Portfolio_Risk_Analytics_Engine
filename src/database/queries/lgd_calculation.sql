-- =============================================================
-- Loss Given Default (LGD) Calculation by Grade
-- LGD = 1 - (total_pymnt / funded_amnt)
-- Only considers defaulted loans with funded_amnt > 0
-- =============================================================

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
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY 1 - total_pymnt / funded_amnt), 4
    )                                                               AS p75_lgd,
    ROUND(
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY 1 - total_pymnt / funded_amnt), 4
    )                                                               AS p25_lgd
FROM loans_master
WHERE loan_status IN (
    'Charged Off',
    'Default',
    'Does not meet the credit policy. Status:Charged Off'
)
AND funded_amnt > 0
GROUP BY grade
ORDER BY grade;
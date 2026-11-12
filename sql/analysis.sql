-- Orrin Digital: people questions, answered in SQL (SQLite) on data_clean/orrin.db.
-- employee_month has one row per person per month employed (point-in-time state from the effective-dated history).
-- Annual resignation rate = resignations / person-years (person-months / 12): the usual "leavers / average headcount".
-- Voluntary = resignations. Dismissals and redundancies are involuntary and are kept out of driver analysis.
-- Run all: python scripts/run_sql.py  (writes sql/results.md)

-- name: 01_workforce_summary
-- How big is the workforce, and how many people joined and left each year?
SELECT CAST(year AS TEXT) AS year,
       ROUND(COUNT(*) / 12.0, 0)                                              AS avg_headcount,
       SUM(m % 12 = 0)                                                        AS headcount_1_jan,
       (SELECT COUNT(*) FROM employees e WHERE strftime('%Y', e.hire_date) = CAST(p.year AS TEXT)) AS hires,
       SUM(terminated)                                                        AS leavers,
       SUM(terminated AND termination_type = 'Resignation')                   AS resignations,
       SUM(terminated AND termination_type IN ('Dismissal', 'Redundancy'))    AS involuntary,
       ROUND(1200.0 * SUM(terminated AND termination_type = 'Resignation') / COUNT(*), 1) AS voluntary_attrition_pct,
       ROUND(1200.0 * SUM(terminated) / COUNT(*), 1)                          AS total_attrition_pct
FROM employee_month p GROUP BY year;

-- name: 02_attrition_by_department
-- Where do people resign? (annual voluntary rate by department and year, resignation month)
SELECT department,
       ROUND(1200.0 * SUM(CASE WHEN year = 2023 THEN resigned END) / SUM(year = 2023), 1) AS rate_2023_pct,
       ROUND(1200.0 * SUM(CASE WHEN year = 2024 THEN resigned END) / SUM(year = 2024), 1) AS rate_2024_pct,
       ROUND(1200.0 * SUM(CASE WHEN year = 2025 THEN resigned END) / SUM(year = 2025), 1) AS rate_2025_pct,
       SUM(resigned)                                                         AS resignations_3y,
       ROUND(AVG(CASE WHEN year = 2025 THEN overtime_3m END), 1)             AS avg_overtime_2025_h
FROM employee_month GROUP BY department ORDER BY rate_2025_pct DESC;

-- name: 03_hiring_freeze
-- What did the Jan-Jun 2025 hiring freeze do to vacancies, overtime and resignations? (by quarter)
WITH hc AS (SELECT month, COUNT(*) AS working, AVG(overtime_3m > 20) AS ot_high, SUM(resigned) AS resignations,
                   SUM(1 - on_notice) AS at_risk
            FROM employee_month GROUP BY month),
     bu AS (SELECT month, SUM(budget_fte) AS budget FROM headcount_budget GROUP BY month)
SELECT strftime('%Y', hc.month) || '-Q' || ((CAST(strftime('%m', hc.month) AS INT) + 2) / 3) AS quarter,
       ROUND(100.0 * AVG(1.0 * (bu.budget - hc.working) / bu.budget), 1)   AS vacancy_rate_pct,
       ROUND(100.0 * AVG(hc.ot_high), 1)                                   AS staff_on_high_overtime_pct,
       SUM(hc.resignations)                                                AS resignations,
       ROUND(1200.0 * SUM(hc.resignations) / SUM(hc.at_risk), 1)           AS annual_resignation_rate_pct
FROM hc JOIN bu USING (month) GROUP BY quarter ORDER BY quarter;

-- name: 04_tenure
-- Do new joiners leave faster? (resignation rate by tenure band)
SELECT CASE WHEN tenure_months < 6 THEN '0-5 months' WHEN tenure_months < 12 THEN '6-11 months'
            WHEN tenure_months < 24 THEN '1-2 years' WHEN tenure_months < 60 THEN '2-5 years'
            WHEN tenure_months < 96 THEN '5-8 years' ELSE '8+ years' END    AS tenure_band,
       COUNT(*)                                                            AS person_months,
       SUM(resigned)                                                       AS resignations,
       ROUND(1200.0 * SUM(resigned) / COUNT(*), 1)                         AS annual_resignation_rate_pct
FROM employee_month WHERE on_notice = 0
GROUP BY tenure_band ORDER BY MIN(tenure_months);

-- name: 05_pay_position
-- Does being paid below the band midpoint matter? (compa-ratio = salary / band midpoint)
SELECT CASE WHEN compa_ratio < 0.85 THEN '1. under 0.85' WHEN compa_ratio < 0.90 THEN '2. 0.85-0.90'
            WHEN compa_ratio < 0.95 THEN '3. 0.90-0.95' WHEN compa_ratio < 1.05 THEN '4. 0.95-1.05'
            ELSE '5. 1.05 and over' END                                    AS compa_band,
       COUNT(*)                                                            AS person_months,
       ROUND(1200.0 * SUM(resigned) / COUNT(*), 1)                         AS annual_resignation_rate_pct,
       ROUND(AVG(tenure_months) / 12.0, 1)                                 AS avg_tenure_years
FROM employee_month WHERE on_notice = 0 AND level <= 4
GROUP BY compa_band ORDER BY compa_band;

-- name: 06_promotion
-- Do people who have waited 3+ years for a promotion leave more? (levels 1-3)
SELECT CASE WHEN months_since_promotion < 18 THEN '0-17 months' WHEN months_since_promotion < 36 THEN '18-35 months'
            WHEN months_since_promotion < 60 THEN '3-5 years' ELSE '5+ years' END AS since_last_promotion,
       COUNT(*)                                                            AS person_months,
       ROUND(1200.0 * SUM(resigned) / COUNT(*), 1)                         AS annual_resignation_rate_pct
FROM employee_month WHERE on_notice = 0 AND level <= 3
GROUP BY since_last_promotion ORDER BY MIN(months_since_promotion);

-- name: 07_overtime
-- Does sustained overtime push people out? (3-month average overtime)
SELECT CASE WHEN overtime_3m IS NULL THEN '(no timesheet)' WHEN overtime_3m < 10 THEN '1. under 10 h'
            WHEN overtime_3m < 20 THEN '2. 10-20 h' WHEN overtime_3m < 30 THEN '3. 20-30 h' ELSE '4. 30 h and over' END AS overtime_per_month,
       COUNT(*)                                                            AS person_months,
       ROUND(1200.0 * SUM(resigned) / COUNT(*), 1)                         AS annual_resignation_rate_pct,
       ROUND(AVG(absence_days), 2)                                         AS absence_days_per_month
FROM employee_month WHERE on_notice = 0 AND level < 5
GROUP BY overtime_per_month ORDER BY overtime_per_month;

-- name: 08_regrettable
-- How many leavers were people we wanted to keep (last rating 4 or 5)?
SELECT CAST(year AS TEXT) AS year,
       SUM(resigned)                                                       AS resignations,
       SUM(resigned AND last_rating >= 4)                                  AS regrettable,
       ROUND(100.0 * SUM(resigned AND last_rating >= 4) / SUM(resigned), 1) AS regrettable_share_pct,
       ROUND(100.0 * AVG(last_rating >= 4), 1)                             AS workforce_rated_4_5_pct
FROM employee_month WHERE last_rating IS NOT NULL GROUP BY year;

-- name: 09_manager_hotspots
-- Which line managers lose people fastest? (teams with 120+ person-months, top 12)
SELECT p.manager_id, m.first_name || ' ' || m.last_name AS manager, m.department,
       COUNT(*)                                                            AS person_months,
       SUM(p.resigned)                                                     AS resignations,
       ROUND(1200.0 * SUM(p.resigned) / COUNT(*), 1)                       AS annual_resignation_rate_pct,
       ROUND(AVG(p.manager_support), 2)                                    AS avg_manager_support_score
FROM employee_month p JOIN employees m ON m.employee_id = p.manager_id
WHERE p.on_notice = 0 AND p.level <= 3
GROUP BY p.manager_id HAVING COUNT(*) >= 120
ORDER BY annual_resignation_rate_pct DESC LIMIT 12;

-- name: 10_return_to_office
-- Did the Sep 2024 return-to-office mandate cost us the people it moved? (staff employed on 1 Sep 2024, next 12 months)
WITH base AS (SELECT employee_id, work_mode,
                     CASE WHEN rto_affected = 1 THEN '1. Moved remote -> hybrid'
                          WHEN work_mode = 'Remote' THEN '2. Stayed remote (remote contract)'
                          ELSE '3. Already office or hybrid' END AS grp
              FROM employee_month WHERE m = 20 AND on_notice = 0)
SELECT b.grp, COUNT(DISTINCT b.employee_id)                                AS people,
       SUM(p.resigned)                                                     AS resigned_next_12m,
       ROUND(1200.0 * SUM(p.resigned) / COUNT(*), 1)                       AS annual_resignation_rate_pct,
       ROUND(AVG(p.engagement), 2)                                         AS avg_engagement
FROM base b JOIN employee_month p ON p.employee_id = b.employee_id AND p.m BETWEEN 20 AND 31 AND p.on_notice = 0
GROUP BY b.grp ORDER BY b.grp;

-- name: 11_new_hire_retention
-- How many new joiners are still here after 6 and 12 months? (hire cohorts 2023-2024)
SELECT strftime('%Y', hire_date) || '-H' || ((CAST(strftime('%m', hire_date) AS INT) + 5) / 6) AS hire_cohort,
       COUNT(*)                                                            AS hires,
       ROUND(100.0 * AVG(termination_date IS NULL OR julianday(termination_date) - julianday(hire_date) >= 182), 1) AS still_here_6m_pct,
       ROUND(100.0 * AVG(termination_date IS NULL OR julianday(termination_date) - julianday(hire_date) >= 365), 1) AS still_here_12m_pct
FROM employees WHERE hire_date >= '2023-01-01' AND hire_date < '2025-01-01'
GROUP BY hire_cohort ORDER BY hire_cohort;

-- name: 12_exit_interviews_vs_data
-- Do exit interviews tell the real story? (stated reason vs the leaver's circumstances in the resignation month)
SELECT x.primary_reason                                                    AS stated_reason,
       COUNT(*)                                                            AS leavers,
       ROUND(100.0 * AVG(p.overtime_3m > 20), 0)                           AS on_high_overtime_pct,
       ROUND(100.0 * AVG(p.compa_ratio < 0.90), 0)                         AS below_band_pct,
       ROUND(100.0 * AVG(p.level <= 3 AND p.months_since_promotion >= 36), 0) AS promotion_stalled_pct
FROM exit_interviews x JOIN employee_month p ON p.employee_id = x.employee_id AND p.resigned = 1
GROUP BY x.primary_reason ORDER BY leavers DESC;

-- name: 13_survey_silence
-- Are the people who stop answering the engagement survey a warning sign?
SELECT CASE WHEN survey_months_ago IS NULL THEN '1. never answered'
            WHEN survey_months_ago <= 3 THEN '2. answered last quarter'
            ELSE '3. stopped answering' END                                AS survey_status,
       COUNT(*)                                                            AS person_months,
       ROUND(AVG(engagement), 2)                                           AS avg_last_engagement,
       ROUND(1200.0 * SUM(resigned) / COUNT(*), 1)                         AS annual_resignation_rate_pct
FROM employee_month WHERE on_notice = 0 AND m >= 3
GROUP BY survey_status ORDER BY survey_status;

-- name: 14_data_quality
-- What did cleaning fix, and does it match what was injected?
SELECT step, table_name, issue, rows_affected, injected, action FROM dq_log;

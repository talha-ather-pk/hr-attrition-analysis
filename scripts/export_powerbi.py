"""
Export the tables the Power BI report loads (data_clean/powerbi/*.csv).

  dim_month, dim_department, dim_location, dim_employee
  fact_employee_month   one row per person per month (point-in-time), with driver flags and resignation cost
  fact_budget           approved headcount by department and month
  fact_risk             today's flight-risk scores (Dec 2025) with the reasons
  fact_event            job-history events (hire, promotion, pay review, manager / work-mode change, leaving)
  fact_exit             exit interviews with the leaver's circumstances in the resignation month
  driver_odds, driver_attributable, manager_hotspots, risk_deciles, scenarios, cost_by_year, cost_assumptions,
  freeze_month, dq_log, findings (headline numbers from the analysis scripts)
Run: python scripts/export_powerbi.py   (after clean, drivers, risk_model and cost)
"""
from pathlib import Path
import json
import sqlite3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data_clean"
OUT = CLEAN / "powerbi"
OUT.mkdir(exist_ok=True)
con = sqlite3.connect(CLEAN / "orrin.db")
P = pd.read_sql("SELECT * FROM employee_month", con)
E = pd.read_sql("SELECT * FROM employees", con)
H = pd.read_sql("SELECT * FROM job_history", con)
X = pd.read_sql("SELECT * FROM exit_interviews", con)
BU = pd.read_sql("SELECT * FROM headcount_budget", con)
DQ = pd.read_sql("SELECT * FROM dq_log", con)
A = pd.read_csv(CLEAN / "cost_assumptions.csv")
SHARE = A.set_index("level").cost_share_of_salary
LEVEL_NAME = A.set_index("level").level_name
DEPTS = ["Engineering", "Data & AI", "Product & Design", "Delivery", "Customer Support", "Sales & Marketing", "Corporate Services"]
LOCS = ["London", "Manchester", "Edinburgh", "Belfast", "Remote (UK)"]


def save(df, name, gz=False):
    path = OUT / (f"{name}.csv.gz" if gz else f"{name}.csv")
    other = OUT / (f"{name}.csv" if gz else f"{name}.csv.gz")
    other.unlink(missing_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", **({"compression": {"method": "gzip", "mtime": 0}} if gz else {}))
    print(f"{name:<22} {len(df):>7,} rows  {path.stat().st_size / 1e6:5.1f} MB")


# ---------------------------------------------------------------- dimensions
months = pd.date_range("2023-01-01", periods=36, freq="MS")
DM = pd.DataFrame({"month_date": months.date, "year": months.year,
                   "quarter": [f"{d.year} Q{(d.month - 1) // 3 + 1}" for d in months],
                   "period": ["Freeze (Jan-Jun 2025)" if pd.Timestamp("2025-01-01") <= d < pd.Timestamp("2025-07-01")
                              else ("After the freeze" if d >= pd.Timestamp("2025-07-01") else "Before the freeze") for d in months]})
save(DM, "dim_month")
save(pd.DataFrame({"department": DEPTS, "dept_order": range(1, 8)}), "dim_department")
save(pd.DataFrame({"location": LOCS, "loc_order": range(1, 6)}), "dim_location")

last = P.sort_values("m").groupby("employee_id").tail(1).set_index("employee_id")
E["name"] = E.first_name + " " + E.last_name
names = E.set_index("employee_id").name
DE = E[["employee_id", "name", "gender", "hire_date", "termination_date", "termination_type", "employment_status", "is_rehire"]].copy()
DE["department"] = DE.employee_id.map(last.department)
DE["job_title"] = DE.employee_id.map(last.job_title)
DE["level"] = DE.employee_id.map(last.level)
DE["location"] = DE.employee_id.map(last.location)
DE["manager"] = DE.employee_id.map(last.manager_id).map(names)
DE["employee_label"] = DE.name + " · " + DE.employee_id
DE = DE[DE.department.notna()]
save(DE, "dim_employee")

# ---------------------------------------------------------------- fact_employee_month
F = P.copy()
F["month_date"] = pd.to_datetime(F.month).dt.date
F["at_risk"] = 1 - F.on_notice
F["ot_high"] = (F.overtime_3m > 20).astype(int)
F["below_band"] = (F.compa_ratio < 0.90).astype(int)
F["promotion_stalled"] = ((F.level <= 3) & (F.months_since_promotion >= 36)).astype(int)
F["low_engagement"] = (F.engagement < 2.9).astype(int)
F["first_year"] = (F.tenure_months < 12).astype(int)
F["regrettable"] = ((F.resigned == 1) & (F.last_rating >= 4)).astype(int)
F["involuntary"] = ((F.terminated == 1) & F.termination_type.isin(["Dismissal", "Redundancy"])).astype(int)
F["resignation_cost"] = np.where(F.resigned == 1, F.annual_salary * F.level.map(SHARE), 0.0).round(0)
first_m = F.groupby("employee_id").m.transform("min")
hire_m = (pd.to_datetime(F.employee_id.map(E.set_index("employee_id").hire_date)).dt.year - 2023) * 12 + \
    pd.to_datetime(F.employee_id.map(E.set_index("employee_id").hire_date)).dt.month - 1
F["is_hire"] = ((F.m == first_m) & (hire_m == F.m)).astype(int)
F["tenure_band"] = pd.cut(F.tenure_months, [-1, 5, 11, 23, 59, 95, 999],
                          labels=["0-5 months", "6-11 months", "1-2 years", "2-5 years", "5-8 years", "8+ years"]).astype(str)
F["compa_band"] = pd.cut(F.compa_ratio, [0, 0.85, 0.90, 0.95, 1.05, 9],
                         labels=["Under 85%", "85-90%", "90-95%", "95-105%", "Over 105%"], right=False).astype(str)
F["overtime_band"] = pd.cut(F.overtime_3m, [-1, 10, 20, 30, 999], labels=["Under 10 h", "10-20 h", "20-30 h", "30 h +"],
                            right=False).astype(str).replace("nan", "")
F["promo_band"] = np.where(F.level > 3, "", pd.cut(F.months_since_promotion, [-1, 17, 35, 59, 9999],
                                                     labels=["Under 18 months", "18-35 months", "3-5 years", "5+ years"]).astype(str))
F["level_name"] = F.level.map(LEVEL_NAME)
F["survey_status"] = np.where(F.survey_months_ago.isna(), "Never answered",
                              np.where(F.survey_months_ago <= 3, "Answered last quarter", "Stopped answering"))
keep = ["employee_id", "month_date", "department", "job_title", "level", "level_name", "location", "work_mode", "manager_id",
        "annual_salary", "band_mid", "compa_ratio", "overtime_hours", "overtime_3m", "months_since_promotion", "tenure_months",
        "age", "last_rating", "engagement", "workload", "manager_support", "growth", "absence_days", "rto_affected",
        "at_risk", "resigned", "terminated", "termination_type", "involuntary", "regrettable", "is_hire", "resignation_cost",
        "ot_high", "below_band", "promotion_stalled", "low_engagement", "first_year", "tenure_band", "compa_band",
        "overtime_band", "promo_band", "survey_status"]
save(F[keep], "fact_employee_month", gz=True)
BU["month_date"] = pd.to_datetime(BU.month).dt.date
save(BU[["month_date", "department", "budget_fte"]], "fact_budget")

# ---------------------------------------------------------------- flight risk, events, exits
R = pd.read_csv(CLEAN / "flight_risk_current.csv")
R["cost_if_leaves"] = (R.annual_salary * R.level.map(SHARE)).round(0)
R["rank"] = range(1, len(R) + 1)
save(R[["employee_id", "name", "department", "job_title", "level", "location", "manager", "annual_salary", "compa_ratio",
        "overtime_3m", "months_since_promotion", "tenure_months", "last_rating", "risk_score", "risk_band", "reasons",
        "cost_if_leaves", "rank"]], "fact_risk")
EV = H[H.event_type != "Migrated record"][["employee_id", "effective_date", "event_type", "department", "job_title", "level",
                                          "work_mode", "manager_id", "annual_salary"]].copy()
EV["effective_date"] = pd.to_datetime(EV.effective_date).dt.date
EV["manager"] = EV.manager_id.map(names)
save(EV.drop(columns="manager_id"), "fact_event", gz=True)
res = P[P.resigned == 1][["employee_id", "overtime_3m", "compa_ratio", "level", "months_since_promotion", "department"]]
XX = X.merge(res, on="employee_id", how="left")
XX["interview_month"] = pd.to_datetime(XX.interview_date).dt.to_period("M").dt.to_timestamp().dt.date
XX["on_high_overtime"] = (XX.overtime_3m > 20).astype(int)
XX["below_band"] = (XX.compa_ratio < 0.90).astype(int)
XX["promotion_stalled"] = ((XX.level <= 3) & (XX.months_since_promotion >= 36)).astype(int)
save(XX[["employee_id", "interview_month", "department", "primary_reason", "would_return", "comment", "on_high_overtime",
         "below_band", "promotion_stalled"]], "fact_exit")

# ---------------------------------------------------------------- analysis tables
OR = pd.read_csv(CLEAN / "driver_odds.csv")
OR["order"] = range(1, len(OR) + 1)
save(OR, "driver_odds")
AT = pd.read_csv(CLEAN / "driver_attributable.csv")
save(AT, "driver_attributable")
HS = pd.read_csv(CLEAN / "manager_hotspots.csv")
HS = HS[HS.person_months >= 60].copy()
HS["status"] = np.where(HS.hotspot == 1, "Hot spot", "Within range")
save(HS[["manager_id", "manager", "department", "employment_status", "person_months", "resignations", "expected",
         "ratio_shrunk", "excess_resignations", "manager_support", "hotspot", "status", "true_bad_manager"]], "manager_hotspots")
save(pd.read_csv(CLEAN / "risk_test_deciles.csv"), "risk_deciles")
S = pd.read_csv(CLEAN / "scenarios.csv")
S["recommended"] = np.where(S.net_annual_benefit > 0, "Yes", "No")
S["order"] = S.net_annual_benefit.rank(ascending=False).astype(int)
save(S[["order", "scenario", "action", "people", "avoided_resignations", "annual_benefit", "annual_cost", "net_annual_benefit",
        "roi_pct", "recommended"]], "scenarios")
CY = pd.read_csv(CLEAN / "turnover_cost_by_year.csv")
save(CY, "cost_by_year")
save(A, "cost_assumptions")
FZ = pd.read_csv(CLEAN / "freeze.csv")
FZ["month_date"] = [months[int(m)].date() for m in FZ.m]
FZ["vacancy_rate"] = (FZ.vacancies / FZ.budget).round(4)
save(FZ[["month_date", "budget", "working", "vacancies", "vacancy_rate"]], "freeze_month")
save(DQ, "dq_log")

# ---------------------------------------------------------------- findings (headline numbers)
dm = json.loads((CLEAN / "driver_metrics.json").read_text())
rm = json.loads((CLEAN / "risk_metrics.json").read_text())
cm = json.loads((CLEAN / "cost_metrics.json").read_text())
lr, ce = rm["test"]["Logistic regression"], rm["test"]["True formula (ceiling)"]
FD = [("auc", lr["auc"]), ("auc_ceiling", ce["auc"]), ("top_decile_capture", lr["top_decile_capture_pct"] / 100),
      ("top_decile_capture_ceiling", ce["top_decile_capture_pct"] / 100), ("top_decile_precision", lr["top_decile_precision_pct"] / 100),
      ("base_rate", lr["base_rate_pct"] / 100), ("lift", lr["top_decile_lift"]),
      ("expected_resignations_12m", rm["current"]["expected_resignations_next_12m"]),
      ("high_risk", rm["current"]["high_risk"]), ("high_risk_regrettable", rm["current"]["high_risk_regrettable"]),
      ("hotspots", dm["manager_hotspots"]["hotspots_flagged"]), ("hotspot_precision", dm["manager_hotspots"]["precision"]),
      ("hotspot_recall", dm["manager_hotspots"]["recall"]), ("hotspot_excess", dm["manager_hotspots"]["excess_resignations_in_hotspots"]),
      ("freeze_saved", cm["freeze_salary_saved"]), ("freeze_overtime", cm["freeze_extra_overtime_paid"]),
      ("freeze_resignation_cost", cm["freeze_resignation_cost"]), ("freeze_net", cm["freeze_net_saving"]),
      ("package_cost", cm["recommended_annual_cost"]), ("package_avoided", cm["recommended_avoided_resignations"]),
      ("package_net", cm["recommended_net_benefit"]), ("pseudo_r2", dm["pseudo_r2"])]
save(pd.DataFrame(FD, columns=["key", "value"]), "findings")

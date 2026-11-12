"""
Clean the raw HRIS extracts and load them into SQLite (data_clean/orrin.db).

Every fix is logged in dq_log with the rows it touched next to the number the generator injected
(data_raw/true_effects.json), so the cleaning can be audited.

It also rebuilds employee_month: one row per person per month they were employed (Jan 2023 - Dec 2025), with the
point-in-time state taken from the effective-dated job history (department, level, salary, band, manager, work mode),
plus overtime, last rating, last survey answer and what happened that month. This is the base for headcount,
attrition rates and the driver and risk models.

Tables: employees, job_history, salary_bands, performance_reviews, engagement_survey, timesheets, absence,
        exit_interviews, headcount_budget, employee_month, dq_log
Run: python scripts/clean.py
"""
from pathlib import Path
import json
import sqlite3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, CLEAN = ROOT / "data_raw", ROOT / "data_clean"
CLEAN.mkdir(exist_ok=True)
DB = CLEAN / "orrin.db"
inj = json.loads((RAW / "true_effects.json").read_text())["injected_mess"]
log = []


def note(table, issue, rows, action, key=None):
    want = inj.get(key) if key else None
    if isinstance(want, list):
        want = len(want)
    log.append({"step": len(log) + 1, "table_name": table, "issue": issue, "rows_affected": int(rows), "action": action,
                "injected": want})
    print(f"{len(log):>2}. {table:<18} {issue:<52} {rows:>7,}  {action}")
    if key:
        assert int(rows) == want, f"{issue}: fixed {rows}, injected {want}"


DEPT = {"Engineering": "Engineering", "Eng": "Engineering", "engineering": "Engineering", "Engineering ": "Engineering",
        "Data & AI": "Data & AI", "Data and AI": "Data & AI", "Data&AI": "Data & AI",
        "Product & Design": "Product & Design", "Product and Design": "Product & Design",
        "Delivery": "Delivery", "Delivery / PMO": "Delivery",
        "Customer Support": "Customer Support", "Support": "Customer Support", "Cust. Support": "Customer Support",
        "Sales & Marketing": "Sales & Marketing", "Sales and Marketing": "Sales & Marketing", "S&M": "Sales & Marketing",
        "Corporate Services": "Corporate Services", "Corporate": "Corporate Services", "Corp Services": "Corporate Services"}


def mi(d):
    """month index: Jan 2023 = 0"""
    d = pd.to_datetime(d)
    return (d.dt.year - 2023) * 12 + d.dt.month - 1


# ------------------------------------------------------------------ job history (the HRIS source of truth)
H = pd.read_csv(RAW / "job_history.csv", dtype=str)
note("job_history", "Rows received", len(H), "loaded")
k = len(H); H = H.drop_duplicates(); note("job_history", "Exact duplicate rows (event posted twice)", k - len(H), "removed", "job_history_duplicate_rows")
bad = ~H.department.isin(set(DEPT.values()))
note("job_history", "Department spelled several ways", bad.sum(), "mapped to the 7 official names", "job_history_department_variants")
H["department"] = H.department.map(DEPT)
assert H.department.notna().all()
H["effective_date"] = pd.to_datetime(H.effective_date)
H["level"] = H.level.astype(int)
H["annual_salary"] = H.annual_salary.astype(float)
H = H.sort_values(["employee_id", "effective_date"], kind="stable").reset_index(drop=True)

# ------------------------------------------------------------------ employees
E = pd.read_csv(RAW / "employees.csv", dtype=str, keep_default_na=False)
note("employees", "Rows received", len(E), "loaded")
test = E.employee_id.str.startswith("TEST")
note("employees", "Test / sandbox records", test.sum(), "removed", "test_records")
E = E[~test].copy()
bad = ~E.department.isin(set(DEPT.values()))
note("employees", "Department spelled several ways", bad.sum(), "mapped to the 7 official names", "employees_department_variants")
E["department"] = E.department.map(DEPT)
dmy = E.hire_date.str.contains("/")
note("employees", "Hire date as dd/mm/yyyy (rest ISO)", dmy.sum(), "parsed day-first", "hire_date_ddmmyyyy")
E["hire_date"] = pd.to_datetime(E.hire_date.where(~dmy), format="%Y-%m-%d").fillna(pd.to_datetime(E.hire_date.where(dmy), format="%d/%m/%Y"))
G = {"Female": "Female", "F": "Female", "female": "Female", "Male": "Male", "M": "Male", "male": "Male",
     "Prefer not to say": "Prefer not to say", "PNTS": "Prefer not to say"}
bad = ~E.gender.isin(["Female", "Male", "Prefer not to say"])
note("employees", "Gender coded several ways (F, female, PNTS ...)", bad.sum(), "standardised", "gender_codes")
E["gender"] = E.gender.map(G)
E["annual_salary"] = E.annual_salary.astype(float)
last = H.groupby("employee_id").tail(1).set_index("employee_id")
hist_sal = E.employee_id.map(last.annual_salary)
monthly = (E.annual_salary * 12 - hist_sal).abs() <= 12
note("employees", "Salary keyed as a monthly figure", monthly.sum(), "x12 (matches job history)", "salary_entered_monthly")
E.loc[monthly, "annual_salary"] = E.loc[monthly, "annual_salary"] * 12
assert ((E.annual_salary - hist_sal).abs() <= 12).all()
stale = (E.termination_date != "") & (E.employment_status == "Active")
note("employees", "Leaver still marked Active", stale.sum(), "set to Terminated", "terminated_but_status_active")
E.loc[stale, "employment_status"] = "Terminated"
blank = (E.manager_id == "") & (E.employee_id.map(last.manager_id).fillna("") != "")
note("employees", "Manager id blank (job history has it)", blank.sum(), "filled from job history", "manager_id_blanked")
E.loc[blank, "manager_id"] = E.loc[blank, "employee_id"].map(last.manager_id)
for c in ("date_of_birth", "resignation_date", "termination_date"):
    E[c] = pd.to_datetime(E[c].replace("", None))
E["level"] = E.level.astype(int)
E["fte"] = E.fte.astype(float)
# rehires ("boomerangs"): same name and date of birth under a new employee id
E = E.sort_values("hire_date").reset_index(drop=True)
E["person_key"] = E.first_name + "|" + E.last_name + "|" + E.date_of_birth.dt.strftime("%Y-%m-%d")
first_id = E.groupby("person_key").employee_id.transform("first")
reh = E.employee_id != first_id
linked = reh & (E.previous_employee_id != "")
note("employees", "Rehired under a new id (same name + date of birth)", reh.sum(),
     f"linked to first id ({linked.sum()} already linked in HRIS)")
E["previous_employee_id"] = np.where(reh, first_id, "")
E["is_rehire"] = reh.astype(int)
E = E.drop(columns="person_key").sort_values("employee_id").reset_index(drop=True)

# ------------------------------------------------------------------ other tables
B = pd.read_csv(RAW / "salary_bands.csv", parse_dates=["effective_from"])
note("salary_bands", "Rows received", len(B), "loaded")
R = pd.read_csv(RAW / "performance_reviews.csv", dtype={"reviewer_id": str}, keep_default_na=False)
note("performance_reviews", "Rows received", len(R), "loaded")
S = pd.read_csv(RAW / "engagement_survey.csv")
note("engagement_survey", "Rows received", len(S), "loaded")
T = pd.read_csv(RAW / "timesheets.csv", parse_dates=["month"])
note("timesheets", "Rows received", len(T), "loaded")
neg = T.overtime_hours < 0
note("timesheets", "Negative overtime (payroll reversal entries)", neg.sum(), "set to missing", "timesheets_negative_overtime")
T.loc[neg, "overtime_hours"] = np.nan
A = pd.read_csv(RAW / "absence.csv", parse_dates=["start_date"])
note("absence", "Rows received", len(A), "loaded")
X = pd.read_csv(RAW / "exit_interviews.csv", parse_dates=["interview_date"])
note("exit_interviews", "Rows received", len(X), "loaded")
BU = pd.read_csv(RAW / "headcount_budget.csv", parse_dates=["month"])
note("headcount_budget", "Rows received", len(BU), "loaded")

# ------------------------------------------------------------------ employee_month (point-in-time panel)
E["hire_m"] = mi(E.hire_date)
E["term_m"] = mi(E.termination_date).fillna(35).astype(int)
E["resign_m"] = mi(E.resignation_date)
rows = []
for r in E.itertuples():
    for m in range(max(int(r.hire_m), 0), int(r.term_m) + 1):
        rows.append((r.employee_id, m))
P = pd.DataFrame(rows, columns=["employee_id", "m"])
P["month"] = P.m.map({m: pd.Timestamp("2023-01-01") + pd.DateOffset(months=m) for m in range(36)})
P["month_end"] = P.month + pd.offsets.MonthEnd(0)
# state as of month end (latest effective-dated record)
st = H[~H.event_type.isin(["Resignation", "Termination", "Retirement notice"])][
    ["employee_id", "effective_date", "department", "job_title", "level", "location", "work_mode", "manager_id", "annual_salary"]]
P = pd.merge_asof(P.sort_values("month_end"), st.sort_values("effective_date"), left_on="month_end",
                  right_on="effective_date", by="employee_id", direction="backward").drop(columns="effective_date")
assert P.department.notna().all()
# pay band in force at month start -> compa-ratio
Bm = B.rename(columns={"effective_from": "band_date"}).sort_values("band_date")
P = pd.merge_asof(P.sort_values("month"), Bm[["department", "level", "location", "band_date", "band_mid"]],
                  left_on="month", right_on="band_date", by=["department", "level", "location"], direction="backward").drop(columns="band_date")
P["compa_ratio"] = (P.annual_salary / P.band_mid).round(3)
# last promotion (job history promotions, else the migrated date, else hire)
promo = H[H.event_type == "Promotion"][["employee_id", "effective_date"]].rename(columns={"effective_date": "promo_date"})
mig = H[H.event_type == "Migrated record"].set_index("employee_id").last_promotion_date
P = pd.merge_asof(P.sort_values("month_end"), promo.sort_values("promo_date"), left_on="month_end", right_on="promo_date",
                  by="employee_id", direction="backward")
base = pd.to_datetime(P.employee_id.map(mig)).fillna(P.employee_id.map(E.set_index("employee_id").hire_date))
P["promo_date"] = P.promo_date.fillna(base)
P["months_since_promotion"] = P.m - mi(P.promo_date)
emp = E.set_index("employee_id")
P["tenure_months"] = P.m - P.employee_id.map(emp.hire_m)
P["age"] = (2023 + P.m / 12 - P.employee_id.map(emp.date_of_birth).dt.year).round(1)
P["gender"] = P.employee_id.map(emp.gender)
P["fte"] = P.employee_id.map(emp.fte)
# overtime: average of this and the previous 2 months (missing timesheets skipped)
T["m"] = mi(T.month)
P = P.merge(T[["employee_id", "m", "overtime_hours"]], on=["employee_id", "m"], how="left")
P = P.sort_values(["employee_id", "m"]).reset_index(drop=True)
P["timesheet_missing"] = P.overtime_hours.isna().astype(int)
P["overtime_3m"] = P.groupby("employee_id").overtime_hours.transform(lambda s: s.rolling(3, min_periods=1).mean()).round(1)
# last rating: the December review of the previous year
R["year_used"] = R.review_year + 1
P["year"] = 2023 + P.m // 12
P = P.merge(R[["employee_id", "year_used", "rating"]].rename(columns={"year_used": "year", "rating": "last_rating"}),
            on=["employee_id", "year"], how="left")
# last survey answer, carried forward (taken at quarter end, so it is known from the next month)
S["m_known"] = S.survey_quarter.str[:4].astype(int).sub(2023) * 12 + S.survey_quarter.str[-1].astype(int) * 3
P = pd.merge_asof(P.sort_values("m"), S[["employee_id", "m_known", "engagement", "workload", "manager_support", "growth", "enps"]]
                  .sort_values("m_known"), left_on="m", right_on="m_known", by="employee_id", direction="backward")
P["survey_months_ago"] = P.m - P.m_known
P = P.drop(columns="m_known")
# absence days started this month
A["m"] = mi(A.start_date)
P = P.merge(A.groupby(["employee_id", "m"]).days.sum().rename("absence_days").reset_index(), on=["employee_id", "m"], how="left")
P["absence_days"] = P.absence_days.fillna(0).astype(int)
# return-to-office mandate: moved from remote to hybrid on 1 Sep 2024
rto = set(H[(H.event_type == "Work mode change") & (H.effective_date == "2024-09-01")].employee_id)
P["rto_affected"] = (P.employee_id.isin(rto) & (P.m >= 20)).astype(int)
# what happened this month
rm = P.employee_id.map(emp.resign_m)
tt = P.employee_id.map(emp.termination_type)
P["resigned"] = ((rm == P.m) & (tt.isin(["Resignation", ""]) | tt.isna()) & P.employee_id.map(emp.resignation_date).notna()
                 & (tt != "Retirement")).astype(int)
P["on_notice"] = (rm < P.m).fillna(False).astype(int)
P["terminated"] = ((P.employee_id.map(emp.term_m) == P.m) & P.employee_id.map(emp.termination_date).notna()).astype(int)
P["termination_type"] = np.where(P.terminated == 1, tt, "")
P["is_manager"] = (P.level >= 4).astype(int)
P = P.sort_values(["employee_id", "m"]).reset_index(drop=True)
note("employee_month", "Person-months built from effective-dated history", len(P), "one row per person per month")
note("employee_month", "Timesheet missing for the month (not heads)", ((P.level < 5) & P.overtime_hours.isna()).sum() - neg.sum(),
     "overtime averaged over the months that exist", "timesheets_missing")
P = P.drop(columns=["month_end"])

# ------------------------------------------------------------------ load
E = E.drop(columns=["hire_m", "term_m", "resign_m"])
with sqlite3.connect(DB) as con:
    for name, df in [("employees", E), ("job_history", H), ("salary_bands", B), ("performance_reviews", R.drop(columns="year_used")),
                     ("engagement_survey", S.drop(columns="m_known")), ("timesheets", T.drop(columns="m")),
                     ("absence", A.drop(columns="m")), ("exit_interviews", X), ("headcount_budget", BU),
                     ("employee_month", P), ("dq_log", pd.DataFrame(log))]:
        out = df.copy()
        for c in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[c]):
                out[c] = out[c].dt.strftime("%Y-%m-%d")
        out.to_sql(name, con, if_exists="replace", index=False)
    con.execute("CREATE INDEX ix_em ON employee_month(employee_id, m)")
pd.DataFrame(log).to_csv(CLEAN / "dq_log.csv", index=False)
print(f"wrote {DB.name}: {len(E):,} employees, {len(P):,} person-months")

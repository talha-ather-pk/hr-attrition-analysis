"""
What does attrition cost, and which fix pays back?

Cost of one resignation = recruitment + vacancy cover + ramp-up of the replacement + manager and team time, each a
share of the leaver's salary (assumptions below, the same ones the Excel template exposes as inputs).

Avoided resignations use the driver model: among people with a flag, the share of resignations the flag causes is
(OR - 1) / OR (attributable fraction among the exposed; a good approximation when the monthly rate is small).
Scenarios are costed on the 2025 run-rate and today's (Dec 2025) workforce.

Outputs (data_clean/): turnover_cost_by_year.csv, turnover_cost_by_dept.csv, cost_assumptions.csv, scenarios.csv,
         freeze.csv, cost_metrics.json
Run: python scripts/cost.py   (after drivers.py)
"""
from pathlib import Path
import json
import sqlite3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data_clean"
con = sqlite3.connect(CLEAN / "orrin.db")
P = pd.read_sql("SELECT * FROM employee_month", con)
OR = pd.read_csv(CLEAN / "driver_odds.csv").set_index("term").odds_ratio
HS = pd.read_csv(CLEAN / "manager_hotspots.csv")

# ---------------------------------------------------------------- cost of one resignation (share of annual salary)
A = pd.DataFrame({
    "level": [1, 2, 3, 4, 5],
    "level_name": ["Associate", "Professional", "Senior", "Manager", "Head"],
    "recruitment_pct": [12, 15, 20, 25, 30],          # agency fee / advertising / assessment
    "vacancy_months": [2, 2, 2.5, 3, 4],              # months the seat is empty
    "vacancy_cover_pct": [50, 50, 50, 50, 50],        # of salary for those months (overtime, contractors, lost output)
    "ramp_months": [3, 4, 6, 6, 9],                   # months for the new starter to reach full output
    "ramp_loss_pct": [50, 50, 50, 50, 50],            # output lost during ramp-up
    "team_time_pct": [4, 5, 5, 6, 8],                 # interviews, onboarding, handover
})
A["cost_share_of_salary"] = ((A.recruitment_pct + A.vacancy_months / 12 * A.vacancy_cover_pct
                              + A.ramp_months / 12 * A.ramp_loss_pct + A.team_time_pct) / 100).round(3)
A.to_csv(CLEAN / "cost_assumptions.csv", index=False)
SHARE = A.set_index("level").cost_share_of_salary
ON_COST = 0.15                                        # employer NI + pension on top of salary rises

P["resignation_cost"] = np.where(P.resigned == 1, P.annual_salary * P.level.map(SHARE), 0.0)
R = P[P.resigned == 1]
by_year = P.groupby("year").agg(avg_headcount=("m", lambda s: len(s) / 12), resignations=("resigned", "sum"),
                                cost=("resignation_cost", "sum"), payroll=("annual_salary", lambda s: s.sum() / 12)).reset_index()
by_year["cost_m"] = (by_year.cost / 1e6).round(2)
by_year["cost_per_resignation"] = (by_year.cost / by_year.resignations).round(0)
by_year["cost_pct_of_payroll"] = (100 * by_year.cost / by_year.payroll).round(1)
by_year.round(1).to_csv(CLEAN / "turnover_cost_by_year.csv", index=False)
print(by_year.round(1).to_string(index=False))
by_dept = R[R.year == 2025].groupby("department").agg(resignations=("resigned", "sum"), cost=("resignation_cost", "sum")).reset_index()
by_dept["cost_m"] = (by_dept.cost / 1e6).round(2)
by_dept.sort_values("cost", ascending=False).round(0).to_csv(CLEAN / "turnover_cost_by_dept.csv", index=False)

# ---------------------------------------------------------------- scenarios (annual, 2025 run-rate)
y25 = P[(P.year == 2025) & (P.on_notice == 0) & (P.level < 5)].copy()
y25["below_band"] = y25.compa_ratio < 0.90
y25["stalled"] = (y25.level <= 3) & (y25.months_since_promotion >= 36)
y25["ot_high"] = y25.overtime_3m > 20
now = P[(P.m == 35) & (P.on_notice == 0) & (P.terminated == 0) & (P.level < 5)].copy()
avg_cost = R[R.year == 2025].resignation_cost.mean()


def avoided(mask, term, reduction=1.0):
    """resignations in 2025 among exposed person-months x attributable fraction x share of the problem removed"""
    res = y25.loc[mask, "resigned"].sum()
    af = (OR[term] - 1) / OR[term]
    cost = y25.loc[mask & (y25.resigned == 1), "resignation_cost"].mean()
    lv = y25.loc[mask & (y25.resigned == 1)].groupby("level").annual_salary.sum().reindex([1, 2, 3, 4], fill_value=0)
    return {**{f"leaver_salary_L{k}": float(v) for k, v in lv.items()}, "leavers": int(res),
            "exposed_resignations": int(res), "odds_ratio": float(OR[term]), "share_removed": float(reduction),
            "avoided_resignations": float(res * af * reduction),
            "cost_per_avoided_resignation": float(cost if pd.notna(cost) else avg_cost)}


rows = []
# 1. pay correction: lift everyone under 0.90 compa to 0.90
gap = (0.90 * now.band_mid - now.annual_salary).clip(lower=0) * (now.compa_ratio < 0.90)
rows.append({"scenario": "Pay correction", "action": "Lift everyone paid under 90 % of band midpoint to 90 %",
             "people": int((gap > 0).sum()), "annual_cost": float(gap.sum() * (1 + ON_COST)), **avoided(y25.below_band, "below_band")})
now.loc[now.compa_ratio < 0.90, ["employee_id", "department", "level", "job_title", "annual_salary", "band_mid", "compa_ratio"]]     .sort_values("compa_ratio").to_csv(CLEAN / "pay_gap_people.csv", index=False)
# 2. promotion: promote stalled people rated 4-5 (10 % rise)
sel = (now.level <= 3) & (now.months_since_promotion >= 36) & (now.last_rating >= 4)
promo_base = float(now.loc[sel, "annual_salary"].sum())
rows.append({"scenario": "Promote stalled top performers", "action": "Promote people rated 4-5 who have waited 3+ years (10 % rise)",
             "people": int(sel.sum()), "annual_cost": promo_base * 0.10 * (1 + ON_COST),
             **avoided(y25.stalled & (y25.last_rating >= 4), "promotion_stalled")})
# 3. workload: end the freeze effect - fill vacancies so overtime stays under 20 h (the 2025 excess over 2024)
ot24 = P[(P.year == 2024) & (P.on_notice == 0) & (P.level < 5)]
excess_share = max(0.0, y25.ot_high.mean() - (ot24.overtime_3m > 20).mean()) / max(y25.ot_high.mean(), 1e-9)
rows.append({"scenario": "Staff to budget (no freeze)", "action": "Keep hiring to approved headcount so overtime stays at 2024 levels",
             "people": None, "annual_cost": None, **avoided(y25.ot_high, "overtime_high", reduction=excess_share)})
# 4. manager hot spots: coaching or re-assignment for flagged managers (assume half the excess goes away)
hs = HS[(HS.hotspot == 1)]
excess_per_year = float(hs.excess_resignations.sum() / 3)
rows.append({"scenario": "Fix manager hot spots", "action": f"Coaching or re-assignment for the {len(hs)} flagged managers (assume half the excess goes)",
             "people": int(len(hs)), "annual_cost": float(len(hs) * 6000),
             **{f"leaver_salary_L{k}": float(v) for k, v in y25[y25.resigned == 1].groupby("level").annual_salary.sum().reindex([1, 2, 3, 4], fill_value=0).items()},
             "leavers": int(y25.resigned.sum()), "exposed_resignations": excess_per_year,
             "odds_ratio": None, "share_removed": 0.5, "avoided_resignations": 0.5 * excess_per_year,
             "cost_per_avoided_resignation": float(avg_cost)})
# 5. return-to-office exceptions for people moved off remote
rows.append({"scenario": "Hybrid exceptions", "action": "Let people moved off remote in Sep 2024 apply for remote exceptions (half take it up)",
             "people": int(now.rto_affected.sum()), "annual_cost": 0.0, **avoided(y25.rto_affected == 1, "rto_mandate", reduction=0.5)})
S = pd.DataFrame(rows)
S["annual_benefit"] = S.avoided_resignations * S.cost_per_avoided_resignation

# the freeze: what it saved in salaries vs what the extra overtime-driven resignations cost
F = P.groupby("m").agg(working=("employee_id", "size"), payroll=("annual_salary", "sum")).reset_index()
bud = pd.read_sql("SELECT month, SUM(budget_fte) AS budget FROM headcount_budget GROUP BY month", con)
F["budget"] = bud.budget.to_numpy()
F["vacancies"] = F.budget - F.working
base_vac = F.loc[F.m.between(12, 23), "vacancies"].mean() / F.loc[F.m.between(12, 23), "budget"].mean()
F["excess_vacancies"] = (F.vacancies - base_vac * F.budget).clip(lower=0)
avg_sal = F.payroll.sum() / F.working.sum()
freeze_saved = float((F.loc[F.m.between(24, 35), "excess_vacancies"] * avg_sal / 12).sum() * (1 + ON_COST))
freeze_cost = float(S.loc[S.scenario == "Staff to budget (no freeze)", "annual_benefit"].iloc[0])
# extra overtime paid in 2025 over the 2024 level (levels 1-3, paid at 1.25 x the hourly rate)
OT_PREMIUM = 1.25
paid = P[(P.level <= 3)].copy()
paid["hourly"] = paid.annual_salary / (12 * 150)
ot_rate24 = paid.loc[paid.year == 2024].groupby("department").overtime_hours.mean()
p25 = paid[paid.year == 2025]
# department average rise in overtime (not person by person, which would count normal month-to-month swings)
extra_per_month = (p25.groupby("department").overtime_hours.mean() - ot_rate24).clip(lower=0)
freeze_overtime = float((p25.department.map(extra_per_month) * p25.hourly * OT_PREMIUM).sum())
S.loc[S.scenario == "Staff to budget (no freeze)", "annual_cost"] = freeze_saved - freeze_overtime
S.loc[S.scenario == "Staff to budget (no freeze)", "action"] += " (cost = salaries the freeze saved less the extra overtime it paid)"
S["net_annual_benefit"] = S.annual_benefit - S.annual_cost
S["roi_pct"] = np.where(S.annual_cost > 0, 100 * S.net_annual_benefit / S.annual_cost, np.nan)
S = S.sort_values("net_annual_benefit", ascending=False)
S.round(4).to_csv(CLEAN / "scenarios.csv", index=False)
F[["m", "budget", "working", "vacancies", "excess_vacancies"]].round(1).to_csv(CLEAN / "freeze.csv", index=False)
print(S[["scenario", "people", "annual_cost", "avoided_resignations", "annual_benefit", "net_annual_benefit", "roi_pct"]].round(0).to_string(index=False))

metrics = {"cost_per_resignation_2025": round(float(avg_cost), 0),
           "turnover_cost_2025_m": float(by_year.loc[by_year.year == 2025, "cost_m"].iloc[0]),
           "turnover_cost_3y_m": round(float(by_year.cost.sum() / 1e6), 2),
           "turnover_cost_pct_payroll_2025": float(by_year.loc[by_year.year == 2025, "cost_pct_of_payroll"].iloc[0]),
           "freeze_salary_saved": round(freeze_saved, 0), "freeze_extra_overtime_paid": round(freeze_overtime, 0),
           "freeze_resignation_cost": round(freeze_cost, 0),
           "freeze_net_saving": round(freeze_saved - freeze_overtime - freeze_cost, 0),
           "recommended": S.loc[S.net_annual_benefit > 0, "scenario"].tolist(),
           "recommended_annual_cost": round(float(S.loc[S.net_annual_benefit > 0, "annual_cost"].sum()), 0),
           "recommended_avoided_resignations": round(float(S.loc[S.net_annual_benefit > 0, "avoided_resignations"].sum()), 0),
           "recommended_net_benefit": round(float(S.loc[S.net_annual_benefit > 0, "net_annual_benefit"].sum()), 0),
           "promotion_salary_base": round(promo_base, 0), "overtime_value_at_1x": round(freeze_overtime / OT_PREMIUM, 0),
           "hotspot_managers": int(len(hs)), "avg_headcount_2025": float(by_year.loc[by_year.year == 2025, "avg_headcount"].iloc[0]),
           "avg_cost_share_of_salary": A.set_index("level_name").cost_share_of_salary.to_dict()}
(CLEAN / "cost_metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=1))

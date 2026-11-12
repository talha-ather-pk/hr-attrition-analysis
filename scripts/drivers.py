"""
What drives resignations? A discrete-time hazard model (logistic regression on person-months at risk), checked
against the generator's truth (data_raw/true_effects.json, read only at the end).

1. Driver model: monthly P(resign) ~ observable flags (overtime, pay vs band, promotion wait, last survey answer,
   tenure, age, rating, return-to-office move) + department, month-of-year and the 2023 tech hiring market.
   Odds ratios with 95 % confidence intervals.
2. Attributable resignations: how many resignations each driver caused (model counterfactual: same people, flag off).
3. Manager hot spots: each line manager's actual resignations vs what the model expects for that team's people.
   "Bad managers" are not in the data; this recovers them from outcomes alone.

Outputs (data_clean/): driver_odds.csv, driver_attributable.csv, manager_hotspots.csv, driver_metrics.json
Run: python scripts/drivers.py
"""
from pathlib import Path
import json
import sqlite3
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data_clean"
con = sqlite3.connect(CLEAN / "orrin.db")
P = pd.read_sql("SELECT * FROM employee_month WHERE on_notice = 0 AND level < 5", con)   # heads: too few to model

DRIVERS = {  # model term: (label, truth key)
    "overtime_high": ("Overtime > 20 h a month (3-month average)", "overtime_high"),
    "below_band": ("Paid under 90 % of band midpoint", "below_band"),
    "promotion_stalled": ("3+ years without promotion (levels 1-3)", "promotion_stalled"),
    "low_engagement": ("Last survey engagement under 2.9", "low_engagement"),
    "rto_mandate": ("Moved remote to hybrid (Sep 2024 mandate)", "rto_mandate"),
    "first_year": ("First year in the job", "first_year"),
}
OTHER = {
    "survey_silent": "Has not answered the survey for 6+ months",
    "long_tenure": "8+ years tenure", "age_under_30": "Under 30", "age_55_plus": "55 and over",
    "manager_level": "Line manager (level 4)",
    "rating_1_2": "Last rating 1-2", "rating_5": "Last rating 5",
}


def features(df):
    X = pd.DataFrame(index=df.index)
    X["overtime_high"] = (df.overtime_3m > 20).astype(int)
    X["below_band"] = (df.compa_ratio < 0.90).astype(int)
    X["promotion_stalled"] = ((df.level <= 3) & (df.months_since_promotion >= 36)).astype(int)
    X["low_engagement"] = (df.engagement < 2.9).astype(int)
    X["rto_mandate"] = df.rto_affected
    X["first_year"] = (df.tenure_months < 12).astype(int)
    X["survey_silent"] = ((df.m >= 6) & (df.survey_months_ago.isna() | (df.survey_months_ago > 6))).astype(int)
    X["long_tenure"] = (df.tenure_months >= 96).astype(int)
    X["age_under_30"] = (df.age < 30).astype(int)
    X["age_55_plus"] = (df.age >= 55).astype(int)
    X["manager_level"] = (df.level == 4).astype(int)
    X["rating_1_2"] = (df.last_rating <= 2).astype(int)
    X["rating_5"] = (df.last_rating == 5).astype(int)
    for d in ["Engineering", "Data & AI", "Customer Support", "Corporate Services", "Product & Design", "Sales & Marketing"]:
        X[f"dept_{d}"] = (df.department == d).astype(int)          # reference: Delivery
    X["tech_market_2023"] = ((df.year == 2023) & df.department.isin(["Engineering", "Data & AI"])).astype(int)
    moy = (df.m % 12) + 1
    for k in [1, 2, 4, 9, 11, 12]:                                   # other months: reference
        X[f"month_{k:02d}"] = (moy == k).astype(int)
    return X


X = features(P)
y = P.resigned
fit = sm.Logit(y, sm.add_constant(X.astype(float))).fit(disp=False)
ci = fit.conf_int()
truth = json.loads((ROOT / "data_raw" / "true_effects.json").read_text())
tc = truth["hazard_model"]["coefficients"]
rows = []
for k in list(DRIVERS) + list(OTHER):
    lab = DRIVERS[k][0] if k in DRIVERS else OTHER[k]
    rows.append({"term": k, "label": lab, "driver": int(k in DRIVERS), "prevalence_pct": round(100 * X[k].mean(), 1),
                 "odds_ratio": round(float(np.exp(fit.params[k])), 2), "ci_low": round(float(np.exp(ci.loc[k, 0])), 2),
                 "ci_high": round(float(np.exp(ci.loc[k, 1])), 2), "p_value": float(fit.pvalues[k]),
                 "true_odds_ratio": round(float(np.exp(tc[k])), 2) if k in tc else None})
OR = pd.DataFrame(rows)
OR["truth_in_ci"] = [None if pd.isna(t) else bool(lo <= t <= hi) for t, lo, hi in zip(OR.true_odds_ratio, OR.ci_low, OR.ci_high)]
OR.to_csv(CLEAN / "driver_odds.csv", index=False)
print(OR[["term", "prevalence_pct", "odds_ratio", "ci_low", "ci_high", "true_odds_ratio", "truth_in_ci"]].to_string(index=False))

# attributable resignations (model counterfactual: flag switched off for the same person-months)
lin = np.asarray(sm.add_constant(X.astype(float)) @ fit.params)
p = 1 / (1 + np.exp(-lin))
att = []
for k, (lab, tk) in DRIVERS.items():
    p0 = 1 / (1 + np.exp(-(lin - fit.params[k] * X[k].to_numpy())))
    est = float((p - p0).sum())
    by_year = {int(yr): round(float((p - p0)[P.year == yr].sum()), 1) for yr in (2023, 2024, 2025)}
    att.append({"driver": k, "label": lab, "attributable_resignations": round(est, 1),
                "share_of_resignations_pct": round(100 * est / y.sum(), 1),
                "true_attributable": truth["attributable_resignations"].get(tk), **{f"att_{yr}": v for yr, v in by_year.items()}})
ATT = pd.DataFrame(att).sort_values("attributable_resignations", ascending=False)
ATT["rank"] = range(1, len(ATT) + 1)
ATT["true_rank"] = ATT.true_attributable.rank(ascending=False).astype(int)
ATT.to_csv(CLEAN / "driver_attributable.csv", index=False)
print(ATT[["driver", "attributable_resignations", "share_of_resignations_pct", "true_attributable", "rank", "true_rank"]].to_string(index=False))

# manager hot spots: observed vs expected resignations among each manager's team (levels 1-3)
P["expected"] = p
team = P[(P.level <= 3) & (P.manager_id.fillna("") != "")]
g = team.groupby("manager_id").agg(person_months=("m", "size"), resignations=("resigned", "sum"), expected=("expected", "sum"),
                                     department=("department", "first"), manager_support=("manager_support", "mean"))
K = 4.0                                                     # shrink small teams toward 1.0 (empirical Bayes style)
g["ratio_raw"] = g.resignations / g.expected
g["ratio_shrunk"] = (g.resignations + K) / (g.expected + K)
emp = pd.read_sql("SELECT employee_id, first_name || ' ' || last_name AS manager, employment_status FROM employees", con).set_index("employee_id")
g = g.join(emp, how="left")
g["excess_resignations"] = (g.resignations - g.expected).round(1)
g["hotspot"] = ((g.ratio_shrunk >= 1.6) & (g.person_months >= 60)).astype(int)
bad = set(truth["bad_managers_ever"])
g["true_bad_manager"] = g.index.isin(bad).astype(int)
g = g.sort_values("ratio_shrunk", ascending=False)
g.round(3).reset_index().rename(columns={"index": "manager_id"}).to_csv(CLEAN / "manager_hotspots.csv", index=False)
big = g[g.person_months >= 60]
tp = int(((big.hotspot == 1) & (big.true_bad_manager == 1)).sum())
hs = {"managers_scored": int(len(big)), "hotspots_flagged": int(big.hotspot.sum()), "true_bad_managers_scored": int(big.true_bad_manager.sum()),
      "flagged_and_truly_bad": tp, "precision": round(float(tp / max(1, big.hotspot.sum())), 2),
      "recall": round(float(tp / max(1, big.true_bad_manager.sum())), 2),
      "excess_resignations_in_hotspots": round(float(big[big.hotspot == 1].excess_resignations.sum()), 1)}
print("manager hot spots:", hs)

metrics = {"person_months_at_risk": int(len(P)), "resignations": int(y.sum()), "pseudo_r2": round(float(fit.prsquared), 3),
           "truth_inside_95ci": f"{int(OR.truth_in_ci.fillna(False).sum())} of {int(OR.truth_in_ci.notna().sum())}",
           "top_three_model": ATT.driver.head(3).tolist(), "top_three_truth": truth["top_three_drivers"],
           "manager_hotspots": hs}
(CLEAN / "driver_metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps({k: v for k, v in metrics.items() if k != "manager_hotspots"}, indent=1))

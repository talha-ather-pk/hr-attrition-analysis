"""
Who is likely to resign in the next 12 months? A flight-risk score with an honest out-of-time test.

Snapshots: every person employed and not on notice at the end of a month, with what HR knew then (overtime, pay vs
band, promotion wait, survey answers and survey silence, absence, rating, tenure, the manager's recent team
resignations). Label: resigned in the following 12 months.

  Test   : train on snapshots Jan - Dec 2023 (labels end Dec 2024), score Dec 2024, compare with 2025 resignations.
  Current: retrain on snapshots Jan 2023 - Dec 2024 (labels end Dec 2025), score everyone at Dec 2025.

Two models: logistic regression (transparent) and gradient boosting; logistic is kept unless boosting is clearly
more accurate. The ceiling is the generator's true formula on the same people (resignation is partly random).
Outputs (data_clean/): risk_test_deciles.csv, risk_metrics.json, flight_risk_current.csv
Run: python scripts/risk_model.py
"""
from pathlib import Path
import json
import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data_clean"
con = sqlite3.connect(CLEAN / "orrin.db")
P = pd.read_sql("SELECT * FROM employee_month", con).sort_values(["employee_id", "m"]).reset_index(drop=True)
EMP = pd.read_sql("SELECT employee_id, first_name || ' ' || last_name AS name FROM employees", con).set_index("employee_id").name

# rolling features per person
P["absence_3m"] = P.groupby("employee_id").absence_days.transform(lambda s: s.rolling(3, min_periods=1).sum())
P["overtime_6m"] = P.groupby("employee_id").overtime_hours.transform(lambda s: s.rolling(6, min_periods=1).mean())
# manager's team: resignations per person-year over the last 12 months (excluding this person)
tm = P[P.level <= 3].groupby(["manager_id", "m"]).agg(pm=("resigned", "size"), res=("resigned", "sum")).reset_index()
tm = tm.sort_values(["manager_id", "m"])
tm["pm12"] = tm.groupby("manager_id").pm.transform(lambda s: s.rolling(12, min_periods=1).sum())
tm["res12"] = tm.groupby("manager_id").res.transform(lambda s: s.rolling(12, min_periods=1).sum())
P = P.merge(tm[["manager_id", "m", "pm12", "res12"]], on=["manager_id", "m"], how="left")
P["team_resign_rate_12m"] = (12 * (P.res12.fillna(0) - P.resigned) / (P.pm12.fillna(0) - 1).clip(lower=6)).fillna(0).round(3)
# label: resigned in the next 12 months
res_m = P[P.resigned == 1].groupby("employee_id").m.min()
P["resign_m"] = P.employee_id.map(res_m)
P["label"] = ((P.resign_m > P.m) & (P.resign_m <= P.m + 12)).astype(int)
snap = P[(P.on_notice == 0) & (P.resigned == 0) & (P.terminated == 0)].copy()

NUM = ["overtime_3m", "overtime_6m", "compa_ratio", "months_since_promotion", "level", "tenure_months", "age", "fte",
       "absence_3m", "rto_affected", "team_resign_rate_12m", "engagement", "workload", "manager_support", "growth", "enps"]
DEPTS = sorted(P.department.unique())


def X(df):
    out = df[NUM].astype(float).copy()
    out["no_rating"] = df.last_rating.isna().astype(int)
    out["last_rating"] = df.last_rating.fillna(3)
    out["survey_silent"] = (df.survey_months_ago.isna() | (df.survey_months_ago > 6)).astype(int)
    for c in ["engagement", "workload", "manager_support", "growth"]:
        out[c] = out[c].fillna(3.5)
    out["enps"] = out.enps.fillna(7)
    out["overtime_3m"] = out.overtime_3m.fillna(out.overtime_3m.median())
    out["overtime_6m"] = out.overtime_6m.fillna(out.overtime_6m.median())
    for d in DEPTS:
        out[f"dept_{d}"] = (df.department == d).astype(int)
    return out


def models():
    return {"Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
            "Gradient boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
                                                                min_samples_leaf=200, l2_regularization=1.0, random_state=7)}


def evaluate(y, s):
    order = np.argsort(-s)
    top = order[: len(s) // 10]
    return {"auc": round(float(roc_auc_score(y, s)), 3), "brier": round(float(brier_score_loss(y, s)), 4),
            "base_rate_pct": round(100 * float(y.mean()), 1),
            "top_decile_precision_pct": round(100 * float(y[top].mean()), 1),
            "top_decile_capture_pct": round(100 * float(y[top].sum() / y.sum()), 1),
            "top_decile_lift": round(float(y[top].mean() / y.mean()), 2)}


train = snap[snap.m <= 11]
test = snap[snap.m == 23]
res, scores = {}, {}
for name, mdl in models().items():
    mdl.fit(X(train), train.label)
    s = mdl.predict_proba(X(test))[:, 1]
    res[name], scores[name] = evaluate(test.label.to_numpy(), s), s
    print(name, res[name])
# prefer the transparent, better-calibrated logistic model unless boosting is clearly more accurate
best = "Gradient boosting" if res["Gradient boosting"]["auc"] > res["Logistic regression"]["auc"] + 0.01 else "Logistic regression"
# ceiling: the generator's true formula applied to the same Dec 2024 flags (truth read only here, for comparison)
truth = json.loads((ROOT / "data_raw" / "true_effects.json").read_text())
c, badm = truth["hazard_model"]["coefficients"], set(truth["bad_managers_ever"])
oracle = (c["overtime_high"] * (test.overtime_3m > 20) + c["below_band"] * (test.compa_ratio < 0.9)
          + c["promotion_stalled"] * ((test.level <= 3) & (test.months_since_promotion >= 36))
          + c["low_engagement"] * (test.engagement < 2.9) + c["bad_manager"] * test.manager_id.isin(badm)
          + c["rto_mandate"] * test.rto_affected + c["first_year"] * (test.tenure_months < 12)
          + c["long_tenure"] * (test.tenure_months >= 96) + c["age_under_30"] * (test.age < 30) + c["manager_level"] * (test.level == 4))
oracle_p = 1 - (1 - 1 / (1 + np.exp(-(truth["hazard_model"]["intercept"] + oracle.to_numpy().astype(float))))) ** 12
res["True formula (ceiling)"] = evaluate(test.label.to_numpy(), oracle_p)
print("True formula (ceiling)", res["True formula (ceiling)"])
s = scores[best]
dec = pd.DataFrame({"score": s, "label": test.label.to_numpy()})
dec["decile"] = 10 - pd.qcut(dec.score.rank(method="first"), 10, labels=False)
TD = dec.groupby("decile").agg(people=("label", "size"), predicted_pct=("score", "mean"), actual_pct=("label", "mean"),
                               resigned=("label", "sum")).reset_index()
TD["predicted_pct"] = (100 * TD.predicted_pct).round(1)
TD["actual_pct"] = (100 * TD.actual_pct).round(1)
TD["cumulative_capture_pct"] = (100 * TD.resigned.cumsum() / TD.resigned.sum()).round(1)
TD.to_csv(CLEAN / "risk_test_deciles.csv", index=False)
print(TD.to_string(index=False))

# current scores: retrain on 2023-2024 snapshots, score everyone at Dec 2025
full = snap[snap.m <= 23]
mdl = models()[best]
mdl.fit(X(full), full.label)
now = P[(P.m == 35) & (P.on_notice == 0) & (P.terminated == 0) & (P.resigned == 0)].copy()
now["risk_score"] = mdl.predict_proba(X(now))[:, 1].round(3)
cut_hi, cut_med = now.risk_score.quantile(0.90), now.risk_score.quantile(0.70)
now["risk_band"] = np.where(now.risk_score >= cut_hi, "High", np.where(now.risk_score >= cut_med, "Medium", "Low"))


def reasons(r):
    out = []
    if r.overtime_3m > 20: out.append(f"Overtime {r.overtime_3m:.0f} h/month")
    if r.compa_ratio < 0.90: out.append(f"Paid {100 * r.compa_ratio:.0f}% of band midpoint")
    if r.level <= 3 and r.months_since_promotion >= 36: out.append(f"{r.months_since_promotion // 12:.0f}+ years since promotion")
    if r.team_resign_rate_12m >= 0.30: out.append("Manager's team losing people fast")
    if r.tenure_months < 12: out.append("First year")
    if r.rto_affected: out.append("Moved off remote (Sep 2024)")
    if pd.notna(r.engagement) and r.engagement < 2.9: out.append("Low engagement score")
    return "; ".join(out) if out else "No single flag (combination of factors)"


now["reasons"] = now.apply(reasons, axis=1)
now["name"] = now.employee_id.map(EMP)
now["manager"] = now.manager_id.map(EMP)
cols = ["employee_id", "name", "department", "job_title", "level", "location", "manager_id", "manager", "annual_salary",
        "compa_ratio", "overtime_3m", "months_since_promotion", "tenure_months", "last_rating", "risk_score", "risk_band", "reasons"]
now = now.sort_values("risk_score", ascending=False)[cols]
now.to_csv(CLEAN / "flight_risk_current.csv", index=False)
metrics = {"chosen_model": best, "test": {"train": "snapshots Jan-Dec 2023", "scored": "Dec 2024", "outcome": "resigned in 2025",
                                          "people": int(len(test)), **{k: v for k, v in res.items()}},
           "current": {"scored_at": "Dec 2025", "people": int(len(now)), "high_risk": int((now.risk_band == "High").sum()),
                       "expected_resignations_next_12m": round(float(now.risk_score.sum()), 0),
                       "high_risk_regrettable": int(((now.risk_band == "High") & (now.last_rating >= 4)).sum()),
                       "high_risk_cut": round(float(cut_hi), 3)}}
(CLEAN / "risk_metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics["current"], indent=1))
print(now.head(8)[["name", "department", "risk_score", "reasons"]].to_string(index=False))

"""
Charts for the README and the summary site (images/chart_*.png), in the dashboard's palette.
Run: python scripts/make_charts.py   (after drivers, risk_model and cost)
"""
from pathlib import Path
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN, IMG = ROOT / "data_clean", ROOT / "images"
IMG.mkdir(exist_ok=True)
INDIGO, LAV, PALE, CORAL, GREEN, RED, INK, MUTED, LINE = ("#3B2F80", "#8B7FD1", "#C9C3E6", "#F2795B", "#2E9E7A", "#C8364B",
                                                          "#1F1B33", "#6B6785", "#E4E0F0")
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10, "axes.edgecolor": LINE, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "semibold", "axes.titlesize": 12, "axes.titlecolor": INK, "axes.titlelocation": "left"})
con = sqlite3.connect(CLEAN / "orrin.db")
P = pd.read_sql("SELECT * FROM employee_month", con)


def save(fig, name):
    fig.savefig(IMG / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# 1 drivers: resignations caused, model vs truth
A = pd.read_csv(CLEAN / "driver_attributable.csv").sort_values("attributable_resignations")
fig, ax = plt.subplots(figsize=(8, 3.8))
y = np.arange(len(A))
ax.barh(y + 0.2, A.attributable_resignations, 0.4, color=INDIGO, label="Model")
ax.barh(y - 0.2, A.true_attributable, 0.4, color=PALE, label="Known truth")
ax.set_yticks(y, A.label)
for i, v in enumerate(A.attributable_resignations):
    ax.text(v + 3, i + 0.2, f"{v:.0f}", va="center", fontsize=9, color=INK)
ax.set_title("Resignations each driver caused, 2023-25 (of 1,313)")
ax.legend(frameon=False, loc="lower right")
ax.set_xlabel("Resignations that would not have happened with the flag switched off")
save(fig, "chart_drivers.png")

# 2 three drivers: resignation rate by band
risk = P[P.on_notice == 0]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
specs = [("Pay position (salary ÷ band midpoint)", pd.cut(risk.compa_ratio, [0, .85, .90, .95, 1.05, 9], right=False,
                                                         labels=["<85%", "85-90%", "90-95%", "95-105%", ">105%"]), risk, [0, 1]),
         ("Overtime, 3-month average", pd.cut(risk.overtime_3m, [-1, 10, 20, 30, 999], right=False,
                                             labels=["<10 h", "10-20 h", "20-30 h", "30 h+"]), risk, [2, 3]),
         ("Since last promotion (levels 1-3)", pd.cut(risk.months_since_promotion, [-1, 17, 35, 59, 9999],
                                                      labels=["<18 m", "18-35 m", "3-5 y", "5+ y"]), risk, [2, 3])]
for ax, (title, band, df, hot) in zip(axes, specs):
    d = df.assign(b=band)
    if "promotion" in title:
        d = d[d.level <= 3]
    r = d.groupby("b", observed=True).resigned.agg(["sum", "size"])
    rate = 1200 * r["sum"] / r["size"]
    ax.bar(range(len(rate)), rate, color=[CORAL if i in hot else INDIGO for i in range(len(rate))])
    ax.set_xticks(range(len(rate)), rate.index.astype(str), fontsize=9)
    for i, v in enumerate(rate):
        ax.text(i, v + 0.6, f"{v:.0f}%", ha="center", fontsize=9, color=INK)
    ax.set_title(title, fontsize=10.5)
axes[0].set_ylabel("Resign per year (%)")
fig.suptitle("The top three drivers, in the raw data", x=0.01, ha="left", fontsize=12, fontweight="semibold", color=INK)
fig.tight_layout()
save(fig, "chart_bands.png")

# 3 freeze: vacancy, overtime and resignations by month
F = pd.read_csv(CLEAN / "freeze.csv")
m = P.groupby("m").agg(ot=("overtime_3m", lambda s: (s > 20).mean()), res=("resigned", "sum"))
x = pd.date_range("2023-01-01", periods=36, freq="MS")
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.axvspan(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30"), color="#FDE7E1", lw=0)
ax.text(pd.Timestamp("2025-01-10"), 47, "Hiring freeze", color=CORAL, fontsize=9, fontweight="semibold")
ax.bar(x, m.res, width=20, color=PALE, label="Resignations")
ax.set_ylabel("Resignations a month")
ax2 = ax.twinx()
ax2.plot(x, 100 * m.ot, color=CORAL, lw=2.4, label="Staff on 20 h+ overtime (%)")
ax2.plot(x, 100 * F.vacancies / F.budget, color=INDIGO, lw=1.8, ls="--", label="Vacancy rate (%)")
ax2.spines["right"].set_visible(True)
ax2.set_ylabel("%")
ax.set_title("The freeze left seats empty; overtime and resignations followed")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=8.5)
save(fig, "chart_freeze.png")

# 4 managers: actual vs expected
H = pd.read_csv(CLEAN / "manager_hotspots.csv")
H = H[H.person_months >= 60]
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ok = H[H.hotspot == 0]; hs = H[H.hotspot == 1]
ax.scatter(ok.expected, ok.resignations, s=18, color=PALE, label="Other managers")
ax.scatter(hs.expected, hs.resignations, s=40, color=CORAL, label="Hot spot (flagged)")
bad = H[(H.true_bad_manager == 1)]
ax.scatter(bad.expected, bad.resignations, s=90, facecolors="none", edgecolors=INDIGO, lw=1.2, label="Truly poor manager (generator)")
lim = max(H.expected.max(), H.resignations.max()) + 1
ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls=":")
ax.set_xlabel("Expected resignations (model, from the team's own risk)")
ax.set_ylabel("Actual resignations, 2023-25")
ax.set_title("Manager hot spots: 11 of 12 flagged are truly poor managers")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
save(fig, "chart_managers.png")

# 5 business case
S = pd.read_csv(CLEAN / "scenarios.csv").sort_values("net_annual_benefit")
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.barh(S.scenario, S.net_annual_benefit / 1e3, color=[GREEN if v > 0 else RED for v in S.net_annual_benefit])
for i, v in enumerate(S.net_annual_benefit / 1e3):
    ax.text(v + (25 if v > 0 else -25), i, f"£{v:,.0f}k", va="center", ha="left" if v > 0 else "right", fontsize=9, color=INK)
ax.axvline(0, color=MUTED, lw=1)
ax.set_xlim(S.net_annual_benefit.min() / 1e3 - 260, S.net_annual_benefit.max() / 1e3 + 300)
ax.set_title("Net benefit a year by option (2025 run-rate)")
ax.set_xlabel("£ thousand")
save(fig, "chart_scenarios.png")

# 6 risk test deciles
D = pd.read_csv(CLEAN / "risk_test_deciles.csv")
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar(D.decile - 0.2, D.predicted_pct, 0.4, color=PALE, label="Predicted")
ax.bar(D.decile + 0.2, D.actual_pct, 0.4, color=INDIGO, label="Actually resigned in 2025")
ax.axhline(D.resigned.sum() / D.people.sum() * 100, color=CORAL, lw=1.2, ls="--", label="Average")
ax.set_xticks(D.decile, [str(d) for d in D.decile])
ax.set_xlabel("Risk decile at Dec 2024 (1 = highest scores)")
ax.set_ylabel("%")
ax.set_title("Out-of-time test: the scores rank people, resignation stays partly random")
ax.legend(frameon=False, fontsize=8.5)
save(fig, "chart_risk.png")

"""
Excel retention business case: live formulas, editable assumptions, the flight-risk list.

Sheets
  Summary          headline numbers and the recommended package (all formulas)
  Inputs           cost of a resignation by level (recruitment, vacancy cover, ramp-up, team time), on-costs,
                   overtime premium, promotion rise, coaching cost, target pay position. Inputs are blue.
  Cost of turnover 2025 resignations by level and what they cost under the Inputs
  Scenarios        five retention options: resignations avoided (driver-model odds ratios), benefit, cost, net, ROI,
                   payback. Change "share of problem removed" or any input and everything recalculates.
  Pay gap          everyone paid under 90 % of band midpoint today, with the rise to the target position
  Flight risk      the people with a High flight-risk score today, why, and what losing them would cost
  Notes            definitions and method

Output: excel/Orrin_retention_business_case.xlsx
Run: python scripts/excel_template.py   (after cost.py and risk_model.py)
"""
from pathlib import Path
import json
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import CellIsRule, DataBarRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.table import Table, TableStyleInfo

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data_clean"
OUT = ROOT / "excel"; OUT.mkdir(exist_ok=True)
INDIGO, NAV, CORAL, INK, MUTED, CANVAS, BORDER = "3B2F80", "2A2160", "F2795B", "1F1B33", "6B6785", "F6F4FA", "E4E0F0"
GOOD, BAD = "D5EFE6", "F7D9DD"
H_FILL = PatternFill("solid", fgColor=INDIGO)
C_FILL = PatternFill("solid", fgColor=CANVAS)
IN_FONT = Font(color="1D4ED8", bold=True)              # inputs in blue, the usual modelling convention
thin = Side(style="thin", color=BORDER)
BOX = Border(top=thin, bottom=thin, left=thin, right=thin)
GBP, GBP0, PCT, NUM1 = '£#,##0;[Red]-£#,##0', '£#,##0', '0.0%', '#,##0.0'

A = pd.read_csv(CLEAN / "cost_assumptions.csv")
S = pd.read_csv(CLEAN / "scenarios.csv")
CM = json.loads((CLEAN / "cost_metrics.json").read_text())
PG = pd.read_csv(CLEAN / "pay_gap_people.csv")
FR = pd.read_csv(CLEAN / "flight_risk_current.csv")
FR = FR[FR.risk_band == "High"].reset_index(drop=True)
import sqlite3
with sqlite3.connect(CLEAN / "orrin.db") as con:
    L25 = pd.read_sql("SELECT level, COUNT(*) AS n, SUM(annual_salary) AS sal FROM employee_month WHERE resigned = 1 AND year = 2025 GROUP BY level", con)

wb = Workbook()


def title(ws, text, sub):
    ws["A1"] = text; ws["A1"].font = Font(bold=True, size=15, color=INDIGO)
    ws["A2"] = sub; ws["A2"].font = Font(italic=True, color=MUTED)
    ws.sheet_view.showGridLines = False


def header(ws, row, labels, widths=None):
    for j, lab in enumerate(labels, 1):
        c = ws.cell(row, j, lab)
        c.font = Font(bold=True, color="FFFFFF"); c.fill = H_FILL
        c.alignment = Alignment(wrap_text=True, vertical="center"); c.border = BOX
    ws.row_dimensions[row].height = 48
    if widths:
        for j, w in enumerate(widths, 1):
            ws.column_dimensions[chr(64 + j)].width = w


def name(nm, ref):
    wb.defined_names[nm] = DefinedName(nm, attr_text=ref)


# ------------------------------------------------------------------ Inputs
inp = wb.active; inp.title = "Inputs"
title(inp, "Inputs", "Blue cells are assumptions: change them and every sheet recalculates.")
header(inp, 4, ["Level", "Level name", "Recruitment (% of salary)", "Vacancy (months empty)", "Vacancy cover (% of salary)",
                "Ramp-up (months)", "Output lost in ramp-up", "Manager & team time (% of salary)", "Cost of one resignation (share of salary)"],
       [8, 14, 15, 13, 15, 12, 13, 17, 20])
for i, r in A.iterrows():
    rr = 5 + i
    inp.cell(rr, 1, int(r.level)); inp.cell(rr, 2, r.level_name)
    for j, v in zip(range(3, 9), [r.recruitment_pct / 100, r.vacancy_months, r.vacancy_cover_pct / 100, r.ramp_months,
                                  r.ramp_loss_pct / 100, r.team_time_pct / 100]):
        c = inp.cell(rr, j, float(v)); c.font = IN_FONT
        c.number_format = "0%" if j in (3, 5, 7, 8) else "0.0"
    c = inp.cell(rr, 9, f"=C{rr}+D{rr}/12*E{rr}+F{rr}/12*G{rr}+H{rr}")
    c.number_format = "0.0%"; c.font = Font(bold=True)
    for j in range(1, 10):
        inp.cell(rr, j).border = BOX
name("CostShare", "Inputs!$I$5:$I$9")
gen = [("On-costs on pay rises (employer NI + pension)", 0.15, "0%", "OnCost"),
       ("Overtime premium (x hourly rate; 0 if overtime is unpaid and not given back as time off)", 1.25, "0.00", "OTPremium"),
       ("Promotion rise", 0.10, "0%", "PromoRise"),
       ("Coaching / re-assignment cost per hot-spot manager (£)", 6000, GBP0, "CoachCost"),
       ("Target pay position (share of band midpoint)", 0.90, "0%", "TargetCompa")]
inp["A12"] = "General"; inp["A12"].font = Font(bold=True, color=INDIGO, size=12)
for k, (lab, v, fmt, nm) in enumerate(gen):
    rr = 13 + k
    inp.cell(rr, 1, lab); inp.merge_cells(start_row=rr, start_column=1, end_row=rr, end_column=6)
    c = inp.cell(rr, 7, v); c.font = IN_FONT; c.number_format = fmt; c.border = BOX
    name(nm, f"Inputs!$G${rr}")
data = [("Average headcount 2025", CM["avg_headcount_2025"], "#,##0", "AvgHeadcount"),
        ("Salary base of stalled people rated 4-5 (£)", CM["promotion_salary_base"], GBP0, "PromoBase"),
        ("Salaries the 2025 freeze saved, incl. on-costs (£)", CM["freeze_salary_saved"], GBP0, "FreezeSaved"),
        ("Extra overtime worked in 2025 vs 2024, valued at 1 x hourly rate (£)", CM["overtime_value_at_1x"], GBP0, "OTValue1x")]
inp["A19"] = "From the data (not assumptions)"; inp["A19"].font = Font(bold=True, color=INDIGO, size=12)
for k, (lab, v, fmt, nm) in enumerate(data):
    rr = 20 + k
    inp.cell(rr, 1, lab); inp.merge_cells(start_row=rr, start_column=1, end_row=rr, end_column=6)
    c = inp.cell(rr, 7, float(v)); c.number_format = fmt; c.border = BOX
    name(nm, f"Inputs!$G${rr}")

# ------------------------------------------------------------------ Cost of turnover
ct = wb.create_sheet("Cost of turnover")
title(ct, "Cost of turnover, 2025", "Resignations in 2025 by level and what they cost under the Inputs.")
header(ct, 4, ["Level", "Level name", "Resignations", "Salary of leavers (£)", "Cost share", "Cost (£)", "Cost per resignation (£)"],
       [8, 14, 14, 20, 12, 16, 22])
for i in range(5):
    rr = 5 + i
    lv = i + 1
    row = L25[L25.level == lv]
    ct.cell(rr, 1, lv); ct.cell(rr, 2, f"=Inputs!B{5 + i}")
    ct.cell(rr, 3, int(row.n.iloc[0]) if len(row) else 0)
    ct.cell(rr, 4, float(row.sal.iloc[0]) if len(row) else 0).number_format = GBP0
    ct.cell(rr, 5, f"=Inputs!I{5 + i}").number_format = "0.0%"
    ct.cell(rr, 6, f"=D{rr}*E{rr}").number_format = GBP0
    ct.cell(rr, 7, f'=IF(C{rr}=0,"",F{rr}/C{rr})').number_format = GBP0
    for j in range(1, 8):
        ct.cell(rr, j).border = BOX
ct["A10"] = "Total"; ct["A10"].font = Font(bold=True)
for col, f, fmt in [("C", "=SUM(C5:C9)", "#,##0"), ("D", "=SUM(D5:D9)", GBP0), ("F", "=SUM(F5:F9)", GBP0), ("G", "=F10/C10", GBP0)]:
    c = ct[f"{col}10"]; c.value = f; c.number_format = fmt; c.font = Font(bold=True); c.border = BOX
ct["A12"] = "Voluntary attrition 2025"; ct["C12"] = "=C10/AvgHeadcount"; ct["C12"].number_format = PCT
ct["A13"] = "Cost as share of the leavers' salaries"; ct["C13"] = "=F10/D10"; ct["C13"].number_format = PCT
for r in (12, 13):
    ct[f"C{r}"].font = Font(bold=True, color=INDIGO)

# ------------------------------------------------------------------ Pay gap
pg = wb.create_sheet("Pay gap")
title(pg, "Pay gap", "Everyone paid under 90 % of band midpoint (Dec 2025). Rise = up to the target position in Inputs.")
header(pg, 4, ["Employee", "Department", "Level", "Job title", "Salary (£)", "Band midpoint (£)", "Pay position", "Rise to target (£)", "New position"],
       [12, 20, 7, 30, 13, 16, 12, 16, 12])
for i, r in PG.iterrows():
    rr = 5 + i
    pg.append([r.employee_id, r.department, int(r.level), r.job_title, float(r.annual_salary), float(r.band_mid),
               f"=E{rr}/F{rr}", f"=MAX(0,TargetCompa*F{rr}-E{rr})", f"=(E{rr}+H{rr})/F{rr}"])
    for j, fmt in [(5, GBP0), (6, GBP0), (7, "0.0%"), (8, GBP0), (9, "0.0%")]:
        pg.cell(rr, j).number_format = fmt
last_pg = 4 + len(PG)
pg.add_table(Table(displayName="PayGap", ref=f"A4:I{last_pg}", tableStyleInfo=TableStyleInfo(name="TableStyleLight15", showRowStripes=True)))
pg.freeze_panes = "A5"
pg.conditional_formatting.add(f"G5:G{last_pg}", DataBarRule(start_type="num", start_value=0.7, end_type="num", end_value=0.9, color=CORAL))

# ------------------------------------------------------------------ Scenarios
sc = wb.create_sheet("Scenarios")
title(sc, "Retention scenarios (annual, 2025 run-rate)",
      "Avoided resignations = resignations among the people affected x (OR - 1) / OR x share of the problem removed.")
cols = ["Scenario", "What we would do", "People", "2025 resignations among them", "Odds ratio (driver model)",
        "Share caused by the driver", "Share of problem removed", "Resignations avoided a year", "Cost per resignation (£)",
        "Annual benefit (£)", "Annual cost (£)", "Net benefit (£)", "ROI", "Payback (months)", "Recommended",
        "helper: leaver salary L1", "L2", "L3", "L4", "leavers"]
header(sc, 4, cols, [26, 48, 9, 13, 11, 12, 12, 12, 13, 14, 14, 14, 9, 11, 13, 13, 13, 13, 13, 9])
cost_f = {"Pay correction": "=SUM(PayGap[Rise to target (£)])*(1+OnCost)",
          "Promote stalled top performers": "=PromoBase*PromoRise*(1+OnCost)",
          "Staff to budget (no freeze)": "=FreezeSaved-OTValue1x*OTPremium",
          "Fix manager hot spots": "=C{r}*CoachCost",
          "Hybrid exceptions": 0}
order = ["Staff to budget (no freeze)", "Pay correction", "Hybrid exceptions", "Fix manager hot spots", "Promote stalled top performers"]
S = S.set_index("scenario").loc[order].reset_index()
for i, r in S.iterrows():
    rr = 5 + i
    people = '=COUNTIF(PayGap[Rise to target (£)],">0")' if r.scenario == "Pay correction" else (None if pd.isna(r.people) else int(r.people))
    sc.cell(rr, 1, r.scenario).font = Font(bold=True)
    sc.cell(rr, 2, r.action).alignment = Alignment(wrap_text=True, vertical="top")
    sc.cell(rr, 3, people)
    sc.cell(rr, 4, round(float(r.exposed_resignations), 1)).number_format = NUM1
    sc.cell(rr, 5, None if pd.isna(r.odds_ratio) else float(r.odds_ratio)).number_format = "0.00"
    sc.cell(rr, 6, f'=IF(E{rr}="",1,(E{rr}-1)/E{rr})').number_format = "0%"
    c = sc.cell(rr, 7, round(float(r.share_removed), 3)); c.font = IN_FONT; c.number_format = "0%"
    sc.cell(rr, 8, f"=D{rr}*F{rr}*G{rr}").number_format = NUM1
    sc.cell(rr, 9, f"=(P{rr}*Inputs!$I$5+Q{rr}*Inputs!$I$6+R{rr}*Inputs!$I$7+S{rr}*Inputs!$I$8)/T{rr}").number_format = GBP0
    sc.cell(rr, 10, f"=H{rr}*I{rr}").number_format = GBP0
    f = cost_f[r.scenario]
    sc.cell(rr, 11, f.format(r=rr) if isinstance(f, str) else f).number_format = GBP
    sc.cell(rr, 12, f"=J{rr}-K{rr}").number_format = GBP
    sc.cell(rr, 13, f'=IF(K{rr}>0,L{rr}/K{rr},"no cost")').number_format = "0%"
    sc.cell(rr, 14, f'=IF(K{rr}<=0,"now",K{rr}/J{rr}*12)').number_format = "0.0"
    sc.cell(rr, 15, f'=IF(L{rr}>0,"Yes","No")')
    for k, lv in enumerate([1, 2, 3, 4]):
        c = sc.cell(rr, 16 + k, float(r[f"leaver_salary_L{lv}"])); c.number_format = GBP0; c.font = Font(color=MUTED)
    c = sc.cell(rr, 20, int(r.leavers)); c.font = Font(color=MUTED)
    sc.row_dimensions[rr].height = 46
    for j in range(1, 21):
        sc.cell(rr, j).border = BOX
        if j != 2:
            sc.cell(rr, j).alignment = Alignment(vertical="top", wrap_text=j == 1)
end = 4 + len(S)
sc.conditional_formatting.add(f"L5:L{end}", CellIsRule(operator="greaterThan", formula=["0"], fill=PatternFill("solid", fgColor=GOOD)))
sc.conditional_formatting.add(f"L5:L{end}", CellIsRule(operator="lessThan", formula=["0"], fill=PatternFill("solid", fgColor=BAD)))
sc.cell(end + 2, 1, "Recommended package").font = Font(bold=True, color=INDIGO, size=12)
for k, (lab, f, fmt) in enumerate([("Annual cost", f'=SUMIF(O5:O{end},"Yes",K5:K{end})', GBP),
                                   ("Resignations avoided a year", f'=SUMIF(O5:O{end},"Yes",H5:H{end})', NUM1),
                                   ("Annual benefit", f'=SUMIF(O5:O{end},"Yes",J5:J{end})', GBP),
                                   ("Net benefit a year", f'=SUMIF(O5:O{end},"Yes",L5:L{end})', GBP)]):
    sc.cell(end + 3 + k, 1, lab)
    c = sc.cell(end + 3 + k, 2, f); c.number_format = fmt; c.font = Font(bold=True); c.alignment = Alignment(horizontal="left")
pkg = end + 3
sc.freeze_panes = "C5"
for col in "PQRST":
    sc.column_dimensions[col].outlineLevel = 1

# ------------------------------------------------------------------ Flight risk
fr = wb.create_sheet("Flight risk")
title(fr, "Flight risk: High band (Dec 2025)",
      "Probability of resigning in the next 12 months. Talk to these people first; the reasons say what to talk about.")
header(fr, 4, ["Employee", "Name", "Department", "Job title", "Level", "Manager", "Salary (£)", "Pay position", "Overtime (h/month)",
               "Months since promotion", "Last rating", "Risk (12 months)", "Why", "Cost if they leave (£)"],
       [11, 20, 18, 28, 7, 20, 12, 10, 11, 12, 9, 11, 70, 15])
for i, r in FR.iterrows():
    rr = 5 + i
    fr.append([r.employee_id, r["name"], r.department, r.job_title, int(r.level), r.manager, float(r.annual_salary),
               float(r.compa_ratio), None if pd.isna(r.overtime_3m) else float(r.overtime_3m), int(r.months_since_promotion),
               None if pd.isna(r.last_rating) else int(r.last_rating), float(r.risk_score), r.reasons,
               f"=G{rr}*INDEX(CostShare,E{rr})"])
    for j, fmt in [(7, GBP0), (8, "0%"), (9, "0.0"), (12, "0%"), (14, GBP0)]:
        fr.cell(rr, j).number_format = fmt
last_fr = 4 + len(FR)
fr.add_table(Table(displayName="FlightRisk", ref=f"A4:N{last_fr}", tableStyleInfo=TableStyleInfo(name="TableStyleLight15", showRowStripes=True)))
fr.freeze_panes = "C5"
fr.conditional_formatting.add(f"L5:L{last_fr}", DataBarRule(start_type="num", start_value=0, end_type="num", end_value=0.8, color=CORAL))

# ------------------------------------------------------------------ Summary
sm = wb.create_sheet("Summary", 0)
title(sm, "Orrin Digital: retention business case", "Synthetic data · FY2025 · all numbers are live formulas on the other sheets")
sm.column_dimensions["A"].width = 44; sm.column_dimensions["B"].width = 18; sm.column_dimensions["C"].width = 4
sm.column_dimensions["D"].width = 30; sm.column_dimensions["E"].width = 16; sm.column_dimensions["F"].width = 12
kpis = [("Average headcount 2025", "=AvgHeadcount", "#,##0"),
        ("Resignations 2025", "='Cost of turnover'!C10", "#,##0"),
        ("Voluntary attrition 2025", "='Cost of turnover'!C12", PCT),
        ("Cost of those resignations", "='Cost of turnover'!F10", GBP0),
        ("Cost per resignation", "='Cost of turnover'!G10", GBP0),
        ("People at High flight risk today", f"=COUNTA(FlightRisk[Employee])", "#,##0"),
        ("Recommended package: annual cost", f"=Scenarios!B{pkg}", GBP),
        ("Recommended package: resignations avoided", f"=Scenarios!B{pkg + 1}", NUM1),
        ("Recommended package: net benefit a year", f"=Scenarios!B{pkg + 3}", GBP)]
sm["A4"] = "Headline"; sm["A4"].font = Font(bold=True, color=INDIGO, size=12)
for k, (lab, f, fmt) in enumerate(kpis):
    rr = 5 + k
    sm.cell(rr, 1, lab).fill = C_FILL
    c = sm.cell(rr, 2, f); c.number_format = fmt; c.font = Font(bold=True, size=12, color=INK); c.fill = C_FILL
    for j in (1, 2):
        sm.cell(rr, j).border = BOX
sm.cell(13, 2).font = Font(bold=True, size=13, color=INDIGO)
sm["D4"] = "Options"; sm["D4"].font = Font(bold=True, color=INDIGO, size=12)
for j, lab in enumerate(["Scenario", "Net benefit a year", "Do it?"], 4):
    c = sm.cell(5, j, lab); c.font = Font(bold=True, color="FFFFFF"); c.fill = H_FILL
for i in range(len(S)):
    rr = 6 + i
    sm.cell(rr, 4, f"=Scenarios!A{5 + i}")
    sm.cell(rr, 5, f"=Scenarios!L{5 + i}").number_format = GBP
    sm.cell(rr, 6, f"=Scenarios!O{5 + i}")
    for j in (4, 5, 6):
        sm.cell(rr, j).border = BOX
sm.conditional_formatting.add(f"E6:E{5 + len(S)}", CellIsRule(operator="greaterThan", formula=["0"], fill=PatternFill("solid", fgColor=GOOD)))
sm.conditional_formatting.add(f"E6:E{5 + len(S)}", CellIsRule(operator="lessThan", formula=["0"], fill=PatternFill("solid", fgColor=BAD)))
ch = BarChart(); ch.type = "bar"; ch.style = 10
ch.legend = None
ch.add_data(Reference(sm, min_col=5, min_row=5, max_row=5 + len(S)), titles_from_data=True)
ch.set_categories(Reference(sm, min_col=4, min_row=6, max_row=5 + len(S)))
ch.series[0].graphicalProperties.solidFill = INDIGO
ch.series[0].graphicalProperties.line.solidFill = INDIGO
ch.series[0].invertIfNegative = False
ch.x_axis.delete = False; ch.y_axis.delete = False
ch.y_axis.majorGridlines = None
ch.y_axis.numFmt = '£#,##0,"k"'; ch.x_axis.scaling.orientation = "maxMin"
ch.x_axis.tickLblPos = "low"; ch.y_axis.crosses = "max"
ch.height, ch.width = 7.5, 17
sm["D12"] = "Net benefit a year by option"; sm["D12"].font = Font(bold=True, color=INDIGO, size=12)
sm.add_chart(ch, "D13")
sm["A16"] = "How to use"; sm["A16"].font = Font(bold=True, color=INDIGO, size=12)
for k, t in enumerate(["1. Check the cost-of-resignation assumptions in Inputs (blue cells).",
                       "2. In Scenarios, set the share of each problem an option removes (blue).",
                       "3. The package above takes every option with a positive net benefit.",
                       "4. Flight risk lists who to talk to first, and why."]):
    sm.cell(17 + k, 1, t).font = Font(color=MUTED)

# ------------------------------------------------------------------ Notes
nt = wb.create_sheet("Notes")
title(nt, "Notes", "Method and definitions")
nt.column_dimensions["A"].width = 30; nt.column_dimensions["B"].width = 110
notes = [("Data", "Synthetic HRIS data for Orrin Digital Ltd (fictional), Jan 2023 - Dec 2025, generated with a known truth."),
         ("Voluntary attrition", "Resignations / average headcount. Dismissals and redundancies are excluded."),
         ("Cost of a resignation", "Recruitment + vacancy cover + ramp-up of the replacement + manager and team time, as shares of the leaver's salary (Inputs)."),
         ("Odds ratio", "From a discrete-time logistic regression of monthly resignations on observable drivers (scripts/drivers.py). OR 2 = twice the odds of resigning in a month, other things equal."),
         ("Share caused by the driver", "(OR - 1) / OR: among people with the driver, the share of their resignations it causes (attributable fraction among the exposed)."),
         ("Staff to budget", "The 2025 hiring freeze saved salaries but raised overtime and resignations. Cost of this option = salaries saved - extra overtime (valued with the overtime premium)."),
         ("Manager hot spots", "Managers whose teams resign far more than the model expects for those people (scripts/drivers.py); 11 of the 12 flagged are truly poor managers in the generator."),
         ("Flight risk", "Logistic model on monthly snapshots; tested on 2025 resignations from Dec 2024 scores (scripts/risk_model.py). Individual resignations are partly random: use the score to prioritise conversations, not to decide."),
         ("Pay gap list", "Holds people under 90 % of midpoint today. Raising TargetCompa above 90 % would also lift people not in this list.")]
for k, (a, b) in enumerate(notes):
    nt.cell(4 + k, 1, a).font = Font(bold=True)
    c = nt.cell(4 + k, 2, b); c.alignment = Alignment(wrap_text=True, vertical="top")
    nt.row_dimensions[4 + k].height = 32

for ws in wb.worksheets:
    ws.sheet_properties.tabColor = CORAL if ws.title == "Summary" else INDIGO
out = OUT / "Orrin_retention_business_case.xlsx"
wb.save(out)
print("wrote", out)

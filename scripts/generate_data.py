"""
Synthetic HR data for "Orrin Digital Ltd" (fictional UK software and IT services company, ~2,300 staff).

A month-by-month workforce simulation, Jan 2023 - Dec 2025. People are hired, reviewed, paid, promoted, managed,
surveyed, and they resign, retire, are dismissed or made redundant. Resignations follow a known monthly hazard
(logistic, with known odds ratios), so every driver the analysis finds can be checked against the truth.

What happens in the story (all known to the generator):
  * Pay bands move with the market every April faster than the pay review, so tenured staff drift below band.
  * Some managers are "bad managers" (their teams resign more and score lower on the survey).
  * Return-to-office mandate, 1 Sep 2024: office-based staff who worked fully remote become hybrid.
  * Hiring freeze Jan - Jun 2025: vacancies stay open, overtime rises, and resignations follow.
  * Redundancy round in Sales & Marketing, Mar 2025 (involuntary, must not count as voluntary attrition).

Outputs (data_raw/):
  employees.csv           HRIS employee master (current or last state), with the mess a real export has
  job_history.csv         effective-dated history: migration, hire, pay review, promotion, manager / work-mode
                          change, resignation, termination
  salary_bands.csv        pay band min / mid / max by department, level, location and effective date
  performance_reviews.csv annual ratings (1-5), December
  engagement_survey.csv   quarterly pulse survey (confidential, linked to employee id), not everyone responds
  timesheets.csv          monthly contracted, worked and overtime hours
  absence.csv             absence episodes
  exit_interviews.csv     stated reason for leaving (voluntary leavers who took part)
  headcount_budget.csv    approved headcount by department and month
  true_effects.json       the true hazard model, attributable resignations, injected-mess counts and more

Run: python scripts/generate_data.py   (seeded; ~20 s)
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
RAW.mkdir(exist_ok=True)
rng = np.random.default_rng(7)

M = 36                                            # months: Jan 2023 (m = 0) to Dec 2025 (m = 35)
BASE = pd.Timestamp("2023-01-01")


def mstart(m):
    return BASE + pd.DateOffset(months=int(m))


def rand_day(m):
    s = mstart(m)
    return s + pd.Timedelta(days=int(rng.integers(0, (mstart(m + 1) - s).days)))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------- organisation
# dept: (initial headcount, L2 pay midpoint GBP Apr 2022, roles, base overtime h/month, hazard shift, remote share, manager title)
DEPTS = {
    "Engineering":        (820, 52000, ["Software Engineer", "QA Engineer", "DevOps Engineer"], 6, 0.15, 0.20, "Engineering Manager"),
    "Data & AI":          (190, 55000, ["Data Analyst", "Data Scientist", "Data Engineer"], 6, 0.20, 0.20, "Data Manager"),
    "Product & Design":   (180, 53000, ["Product Manager", "UX Designer"], 5, 0.00, 0.15, "Product Lead"),
    "Delivery":           (230, 47000, ["Project Manager", "Business Analyst"], 10, 0.00, 0.10, "Delivery Manager"),
    "Customer Support":   (420, 29000, ["Support Analyst", "Service Desk Agent"], 11, 0.10, 0.06, "Support Team Lead"),
    "Sales & Marketing":  (240, 40000, ["Account Executive", "Marketing Specialist"], 4, 0.00, 0.08, "Sales Manager"),
    "Corporate Services": (180, 38000, ["HR Advisor", "Finance Analyst", "Office Coordinator"], 3, -0.20, 0.08, "Operations Manager"),
}
DN = list(DEPTS)
ND = len(DN)
D_HC = np.array([DEPTS[d][0] for d in DN])
D_MID = np.array([DEPTS[d][1] for d in DN], float)
D_OT = np.array([DEPTS[d][3] for d in DN], float)
D_SHIFT = np.array([DEPTS[d][4] for d in DN])
D_REMOTE = np.array([DEPTS[d][5] for d in DN])

LEVELS = {1: "Associate", 2: "", 3: "Senior", 4: "Manager", 5: "Head"}
L_MULT = np.array([0, 0.78, 1.0, 1.28, 1.6, 2.1])
NOTICE = np.array([0, 1, 1, 2, 3, 3])              # notice period, months, by level
LOCS = ["London", "Manchester", "Edinburgh", "Belfast", "Remote (UK)"]
LOC_P = [0.38, 0.27, 0.15, 0.10, 0.10]
LOC_F = np.array([1.15, 1.0, 1.0, 0.92, 1.0])
MODES = ["Office", "Hybrid", "Remote"]
APRILS = {3: 0.06, 15: 0.045, 27: 0.03}            # band midpoint uplift each April (market)
PAY_RISE = {3: 0.04, 15: 0.03, 27: 0.025}          # average pay review each April
RATING_ADJ = {1: -1.0, 2: -0.015, 3: 0.0, 4: 0.015, 5: 0.03}   # -1.0 means no rise
RTO_M, FREEZE = 20, range(24, 30)                  # Sep 2024; Jan - Jun 2025
REDUNDANCY_M, REDUNDANCY_N = 26, 25                # Mar 2025, Sales & Marketing
GROWTH = {2023: 0.08, 2024: 0.05, 2025: 0.0}       # approved headcount growth a year

# true voluntary-resignation hazard (monthly logit). Odds ratios are exp(coef).
A0 = -5.07
COEF = {"overtime_high": 0.88, "below_band": 0.64, "promotion_stalled": 0.47, "low_engagement": 0.47,
        "bad_manager": 0.59, "rto_mandate": 0.53, "first_year": 0.41, "long_tenure": -0.30,
        "age_under_30": 0.20, "age_55_plus": -0.40, "manager_level": -0.25, "head_level": -1.0,
        "rating_1_2": 0.15, "rating_5": -0.10}
DRIVERS = ["overtime_high", "below_band", "promotion_stalled", "low_engagement", "bad_manager", "rto_mandate", "first_year"]
SEASON = np.array([0.30, 0.10, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, -0.05, -0.40])
MARKET = {2023: {"Engineering": 0.25, "Data & AI": 0.25}, 2024: {}, 2025: {}}


def bandf(m):
    f = 1.0
    for am, pct in APRILS.items():
        if m >= am:
            f *= 1 + pct
    return f


def band_mid(d, lvl, loc, m):
    return D_MID[d] * L_MULT[lvl] * LOC_F[loc] * bandf(m)


FIRST = ["Oliver", "Amelia", "Jack", "Isla", "Harry", "Ava", "George", "Mia", "Noah", "Emily", "Leo", "Sophia", "Arthur",
         "Grace", "Muhammad", "Lily", "Oscar", "Freya", "Charlie", "Ella", "Jacob", "Chloe", "Thomas", "Evie", "James",
         "Ruby", "William", "Poppy", "Ethan", "Hannah", "Aarav", "Priya", "Daniel", "Sarah", "Adam", "Zara", "Ryan", "Aisha",
         "Liam", "Niamh", "Callum", "Eilidh", "Sean", "Aoife", "Kwame", "Ama", "Tomasz", "Zofia", "Luca", "Giulia", "Ali",
         "Fatima", "Hassan", "Ayesha", "Chen", "Mei", "Raj", "Anjali", "David", "Rachel", "Michael", "Laura", "Samuel", "Megan"]
LAST = ["Smith", "Jones", "Taylor", "Brown", "Williams", "Wilson", "Johnson", "Davies", "Patel", "Robinson", "Wright",
        "Thompson", "Evans", "Walker", "White", "Roberts", "Green", "Hall", "Wood", "Jackson", "Clarke", "Khan", "Ahmed",
        "Hussain", "Ali", "Singh", "Shah", "Murphy", "O'Brien", "Kelly", "Campbell", "Stewart", "MacDonald", "Scott",
        "Reid", "Morrison", "Nowak", "Kowalski", "Rossi", "Chen", "Wang", "Okafor", "Mensah", "Begum", "Hughes", "Edwards",
        "Turner", "Hill", "Moore", "Cooper", "Ward", "Morris", "King", "Baker", "Harrison", "Lewis", "Young", "Martin"]

# ---------------------------------------------------------------- people (arrays)
CAP = 7000
dept = np.full(CAP, -1); role = np.zeros(CAP, int); level = np.zeros(CAP, int); loc = np.zeros(CAP, int)
mode = np.zeros(CAP, int); team = np.full(CAP, -1); salary = np.zeros(CAP); hire_m = np.zeros(CAP, int)
promo_m = np.zeros(CAP, int); rating = np.full(CAP, np.nan); perf = np.zeros(CAP); eng_lat = np.zeros(CAP)
ot_prop = np.zeros(CAP); status = np.zeros(CAP, int)          # 0 not hired, 1 active, 2 working notice, 3 left
end_m = np.full(CAP, -1); birth = np.zeros(CAP, int); gender = np.zeros(CAP, int); rto = np.zeros(CAP, bool)
bad = np.zeros(CAP, bool); part_time = np.zeros(CAP, bool); fname = np.zeros(CAP, int); lname = np.zeros(CAP, int)
ot_hist = np.full((CAP, 3), np.nan); eng_eps = np.zeros(CAP); eng_now = np.full(CAP, np.nan)
hire_date = {}; resign_date = {}; term_date = {}; term_type = {}; prev_id = {}; true_driver = {}
n = 0

teams_dept, teams_mgr, teams_lead = [], [], []     # team -> dept, manager person index, is leadership team
events = []                                        # job_history rows
reviews, survey, timesheets, absence = [], [], [], []
boomerangs = []


def eid(i):
    return f"OD{10001 + i}"


def title(i):
    d, lv = dept[i], level[i]
    if lv == 5:
        return f"Head of {DN[d]}"
    if lv == 4:
        return DEPTS[DN[d]][6]
    r = DEPTS[DN[d]][2][role[i]]
    return f"{LEVELS[lv]} {r}".strip()


def mgr_of(i):
    t = team[i]
    return teams_mgr[t] if t >= 0 else -1


def log(i, date, event, **extra):
    mg = mgr_of(i)
    events.append({"employee_id": eid(i), "effective_date": date, "event_type": event, "department": DN[dept[i]],
                   "job_title": title(i), "level": int(level[i]), "location": LOCS[loc[i]], "work_mode": MODES[mode[i]],
                   "manager_id": eid(mg) if mg >= 0 and mg != i else "", "annual_salary": round(float(salary[i]), 0),
                   **extra})


def new_person(d, lvl, m, compa=None, ten_m=0, like=None):
    """Create a person. m = month they join (or 0 for initial staff with tenure ten_m)."""
    global n
    i = n
    n += 1
    dept[i], level[i] = d, lvl
    role[i] = rng.integers(len(DEPTS[DN[d]][2]))
    if like is not None:                            # boomerang: same person, new employee id
        fname[i], lname[i], birth[i], gender[i] = fname[like], lname[like], birth[like], gender[like]
        perf[i], eng_lat[i], ot_prop[i] = perf[like], eng_lat[like] + 0.1, ot_prop[like]
        loc[i] = loc[like]
    else:
        gender[i] = rng.choice(3, p=[0.52, 0.46, 0.02])
        fname[i], lname[i] = rng.integers(len(FIRST)), rng.integers(len(LAST))
        age = {1: 23, 2: 28, 3: 34, 4: 39, 5: 45}[lvl] + rng.gamma(2.0, 2.5) + ten_m / 24
        birth[i] = int(2023 + m / 12 - age)
        perf[i] = rng.normal(0.3 if lvl >= 4 else 0, 1)
        eng_lat[i] = rng.normal(3.7, 0.45)
        ot_prop[i] = rng.gamma(2.0, 4.0) - 8.0
        loc[i] = rng.choice(5, p=LOC_P)
    part_time[i] = lvl <= 3 and rng.random() < 0.07
    if loc[i] == 4:
        mode[i] = 2
    elif m < RTO_M and rng.random() < D_REMOTE[d] and lvl <= 4:
        mode[i] = 2
    else:
        mode[i] = 0 if rng.random() < 0.22 else 1
    if compa is None:
        compa = rng.normal(1.0, 0.06)
    salary[i] = round(band_mid(d, lvl, loc[i], m) * compa, -1)
    hire_m[i] = m - ten_m
    status[i] = 1
    return i


def set_team_for_new(i):
    d = dept[i]
    cands = [t for t in range(len(teams_dept)) if teams_dept[t] == d and not teams_lead[t]]
    sizes = [np.sum((team[:n] == t) & np.isin(status[:n], (1, 2))) for t in cands]
    team[i] = cands[int(np.argmin(sizes))]


# initial organisation, 1 Jan 2023
lead_team = {}
for d in range(ND):
    H = D_HC[d]
    n_t = int(round((H - 1) / 10.5))
    head = new_person(d, 5, 0, compa=rng.normal(1.0, 0.05), ten_m=int(rng.integers(36, 140)))
    lt = len(teams_dept); teams_dept.append(d); teams_mgr.append(head); teams_lead.append(True); lead_team[d] = lt
    for _ in range(n_t):
        ten = int(np.clip(rng.gamma(1.6, 36) + 24, 6, 240))
        mi = new_person(d, 4, 0, compa=np.clip(rng.normal(1.0 - 0.012 * ten / 12, 0.07), 0.72, 1.25), ten_m=ten)
        team[mi] = lt
        bad[mi] = rng.random() < 0.10
        teams_dept.append(d); teams_mgr.append(mi); teams_lead.append(False)
    for _ in range(H - 1 - n_t):
        lvl = rng.choice([1, 2, 3], p=[0.30, 0.42, 0.28])
        ten = int(np.clip(rng.gamma(1.3, 30), 0, 240))
        i = new_person(d, lvl, 0, compa=np.clip(rng.normal(1.0 - 0.012 * ten / 12, 0.07), 0.72, 1.25), ten_m=ten)
        set_team_for_new(i)
N0 = n
for i in range(N0):
    ten = -hire_m[i]
    since = ten if level[i] == 1 else int(min(ten, rng.gamma(2.0, 15)))
    promo_m[i] = -since
    hire_date[i] = rand_day(hire_m[i])
    if hire_m[i] <= -3:                                                     # Dec 2022 review
        z = perf[i] * 0.8 + rng.normal(0, 0.6)
        rating[i] = 1 + np.searchsorted([-1.645, -0.842, 0.524, 1.405], z)
        reviews.append({"employee_id": eid(i), "review_year": 2022, "rating": int(rating[i]),
                        "reviewer_id": eid(mgr_of(i)) if mgr_of(i) >= 0 else ""})
for i in range(N0):
    log(i, BASE, "Migrated record", hire_date=hire_date[i], last_promotion_date=mstart(promo_m[i]) if promo_m[i] > hire_m[i] else hire_date[i])

budget = D_HC.astype(float).copy()
budget_rows = []
acc_attr = {k: 0.0 for k in DRIVERS}
acc_expected = 0.0
prev_rows = []                                     # prevalence of drivers among active person-months
survey_invited, survey_low_resp = 0, []
dept_month = []
bad_ever = set(int(i) for i in np.where(bad[:n])[0])


def leave_now(i, m, date, ttype):
    status[i] = 3
    term_date[i], term_type[i] = date, ttype
    log(i, date, "Termination", termination_type=ttype)


def replace_manager(i, m):
    """i left; find a new manager for the team(s) i led."""
    for t in [t for t in range(len(teams_mgr)) if teams_mgr[t] == i]:
        d = teams_dept[t]
        start = min(mstart(m + 1), pd.Timestamp("2025-12-31"))
        if teams_lead[t]:                                                   # head of department: external
            new = new_person(d, 5, m + 1, compa=rng.normal(1.0, 0.05))
            hire_date[new] = start
            promo_m[new] = m + 1
            teams_mgr[t] = new
            log(new, start, "Hire")
        else:
            seniors = np.where((team[:n] == t) & (status[:n] == 1) & (level[:n] == 3))[0]
            if len(seniors) and rng.random() < 0.6:
                new = seniors[np.argmax(np.nan_to_num(rating[seniors], nan=3) + rng.random(len(seniors)) * 0.5)]
                level[new] = 4
                salary[new] = round(max(salary[new] * 1.12, band_mid(d, 4, loc[new], m + 1) * 0.95), -1)
                promo_m[new] = m + 1
                team[new] = lead_team[d]
                teams_mgr[t] = new
                log(new, start, "Promotion")
            else:
                new = new_person(d, 4, m + 1)
                hire_date[new] = start
                promo_m[new] = m + 1
                team[new] = lead_team[d]
                teams_mgr[t] = new
                log(new, start, "Hire")
            bad[new] = rng.random() < 0.10
            if bad[new]:
                bad_ever.add(int(new))
        for j in np.where((team[:n] == t) & np.isin(status[:n], (1, 2)))[0]:
            if j != teams_mgr[t]:
                log(j, start, "Manager change")


# ---------------------------------------------------------------- monthly simulation
for m in range(M):
    year, moy = 2023 + m // 12, m % 12
    act = np.where(status[:n] == 1)[0]

    # April: pay review, then promotions (L1 -> L2, L2 -> L3; L3 -> L4 only through manager vacancies)
    if m in PAY_RISE:
        for i in act:
            if hire_m[i] <= m - 3:
                r = 3 if np.isnan(rating[i]) else int(rating[i])
                rise = 0.0 if RATING_ADJ[r] == -1.0 else max(0.0, PAY_RISE[m] + RATING_ADJ[r] + rng.normal(0, 0.005))
                salary[i] = round(salary[i] * (1 + rise), -1)
                log(i, mstart(m), "Pay review")
        for i in act:
            if level[i] <= 2 and m - promo_m[i] >= 18 and not np.isnan(rating[i]):
                p = 0.06 + 0.18 * (rating[i] >= 4) + 0.16 * (rating[i] == 5) - 0.05 * (rating[i] <= 2)
                if rng.random() < p:
                    level[i] += 1
                    salary[i] = round(max(salary[i] * 1.10, band_mid(dept[i], level[i], loc[i], m) * 0.92), -1)
                    promo_m[i] = m
                    log(i, mstart(m), "Promotion")

    # Sep 2024: return-to-office mandate for office-based staff working fully remote
    if m == RTO_M:
        for i in act:
            if mode[i] == 2 and loc[i] != 4:
                mode[i], rto[i] = 1, True
                log(i, mstart(m), "Work mode change")

    # budget
    budget *= (1 + GROWTH[year]) ** (1 / 12)
    if m == REDUNDANCY_M:
        budget[DN.index("Sales & Marketing")] -= REDUNDANCY_N
    for d in range(ND):
        budget_rows.append({"department": DN[d], "month": mstart(m).date(), "budget_fte": int(round(budget[d]))})

    # staffing gap -> overtime
    working = np.where(np.isin(status[:n], (1, 2)))[0]
    hc = np.bincount(dept[working], minlength=ND)
    gap = np.clip((budget - hc) / budget, 0, None)
    ot = np.clip(D_OT[dept[working]] + 180 * gap[dept[working]] + ot_prop[working] + rng.normal(0, 3, len(working)), 0, None)
    ot[level[working] == 5] *= 0.5
    ot_hist[working] = np.column_stack([ot_hist[working, 1], ot_hist[working, 2], ot])
    for d in range(ND):
        sel = dept[working] == d
        dept_month.append({"month": m, "department": DN[d], "budget": round(float(budget[d]), 1), "working": int(hc[d]),
                           "gap": round(float(gap[d]), 3), "overtime_mean": round(float(ot[sel].mean()), 1),
                           "overtime_high_share": round(float((np.nanmean(ot_hist[working[sel]], axis=1) > 20).mean()), 3)})

    # quarterly engagement noise
    if moy % 3 == 0:
        eng_eps[working] = rng.normal(0, 0.25, len(working))

    # current driver flags for everyone working
    w = working
    ten = m - hire_m[w]
    age = 2023 + m / 12 - birth[w]
    compa = salary[w] / band_mid(dept[w], level[w], loc[w], m)
    F = {
        "overtime_high": np.nanmean(ot_hist[w], axis=1) > 20,
        "below_band": compa < 0.90,
        "promotion_stalled": (level[w] <= 3) & (m - promo_m[w] >= 36),
        "bad_manager": np.array([bad[mgr_of(i)] if mgr_of(i) >= 0 else False for i in w]),
        "rto_mandate": rto[w],
        "first_year": ten < 12,
    }
    eng = (eng_lat[w] - 0.45 * F["overtime_high"] - 0.40 * F["bad_manager"] - 0.30 * F["below_band"]
           - 0.35 * F["rto_mandate"] - 0.20 * F["promotion_stalled"] + eng_eps[w])
    eng_now[w] = eng
    F["low_engagement"] = eng < 2.9

    # survey (Mar, Jun, Sep, Dec): response is less likely when disengaged
    if moy % 3 == 2:
        resp = rng.random(len(w)) < sigmoid(1.3 + 1.0 * (eng - 3.4))
        survey_invited += len(w)
        survey_low_resp.append([float(resp[F["low_engagement"]].mean()), float(resp[~F["low_engagement"]].mean())])
        for k in np.where(resp)[0]:
            i = w[k]
            e = float(np.clip(eng[k] + rng.normal(0, 0.3), 1, 5))
            survey.append({"survey_quarter": f"{year}-Q{moy // 3 + 1}", "employee_id": eid(i),
                           "engagement": round(e, 1),
                           "workload": round(float(np.clip(3.8 - 0.9 * F["overtime_high"][k] + rng.normal(0, 0.5), 1, 5)), 1),
                           "manager_support": round(float(np.clip(3.9 - 1.0 * F["bad_manager"][k] + rng.normal(0, 0.5), 1, 5)), 1),
                           "growth": round(float(np.clip(3.6 - 0.8 * F["promotion_stalled"][k] - 0.3 * F["below_band"][k] + rng.normal(0, 0.5), 1, 5)), 1),
                           "enps": int(np.clip(round(2.4 * e - 1.5 + rng.normal(0, 1.2)), 0, 10))})

    # timesheets and absence
    for k, i in enumerate(w):
        contracted = 120 if part_time[i] else 150
        if level[i] < 5:
            timesheets.append({"employee_id": eid(i), "month": mstart(m).date(), "contracted_hours": contracted,
                               "overtime_hours": round(max(0.0, ot[k] * (0.8 if part_time[i] else 1) + rng.normal(0, 1)), 1)})
    lam = 0.45 + 0.5 * F["overtime_high"] + 0.4 * F["low_engagement"]
    days = rng.poisson(lam)
    for k in np.where(days > 0)[0]:
        absence.append({"employee_id": eid(w[k]), "start_date": rand_day(m).date(), "days": int(days[k]),
                        "reason": rng.choice(["Sickness", "Sickness", "Sickness", "Sickness", "Family", "Unauthorised"])})

    # resignation hazard (only staff not already on notice)
    on = status[w] == 1
    rr = rating[w]
    mkt = np.array([MARKET[year].get(DN[d], 0.0) for d in dept[w]])
    logit = (A0 + D_SHIFT[dept[w]] + SEASON[moy] + mkt
             + sum(COEF[k] * F[k] for k in DRIVERS)
             + COEF["long_tenure"] * (ten >= 96) + COEF["age_under_30"] * (age < 30) + COEF["age_55_plus"] * (age >= 55)
             + COEF["manager_level"] * (level[w] == 4) + COEF["head_level"] * (level[w] == 5)
             + COEF["rating_1_2"] * (rr <= 2) + COEF["rating_5"] * (rr == 5))
    p = sigmoid(logit)
    acc_expected += p[on].sum()
    for k in DRIVERS:
        acc_attr[k] += (p[on] - sigmoid(logit[on] - COEF[k] * F[k][on])).sum()
    prev_rows.append({"month": m, **{k: float(F[k][on].mean()) for k in DRIVERS}})
    resign = on & (rng.random(len(w)) < p)
    for k in np.where(resign)[0]:
        i = w[k]
        contrib = {d_: COEF[d_] * F[d_][k] for d_ in DRIVERS}
        top = max(contrib, key=contrib.get)
        true_driver[i] = top if contrib[top] > 0 else "none"
        status[i] = 2
        resign_date[i] = rand_day(m)
        end_m[i] = m + NOTICE[level[i]]
        term_type[i] = "Resignation"
        log(i, resign_date[i], "Resignation")
    # retirement (age 63+), notice 3 months
    ret = on & ~resign & (age >= 63) & (rng.random(len(w)) < 0.025)
    for k in np.where(ret)[0]:
        i = w[k]
        status[i] = 2; resign_date[i] = rand_day(m); end_m[i] = m + 3; term_type[i] = "Retirement"
        log(i, resign_date[i], "Retirement notice")
    # dismissal (performance)
    pd_ = np.where(rr == 1, 0.025, np.where(rr == 2, 0.003, 0.0004))
    dis = on & ~resign & ~ret & (rng.random(len(w)) < pd_) & (level[w] < 5)
    for k in np.where(dis)[0]:
        i = w[k]
        leave_now(i, m, rand_day(m), "Dismissal")
        replace_manager(i, m)
    # redundancy round
    if m == REDUNDANCY_M:
        sd = DN.index("Sales & Marketing")
        pool = [i for i in w if status[i] == 1 and dept[i] == sd and level[i] <= 3]
        wt = np.array([{1: 4, 2: 3, 3: 1.5, 4: 1, 5: 0.7}[int(rating[i]) if not np.isnan(rating[i]) else 3] for i in pool], float)
        for i in rng.choice(pool, REDUNDANCY_N, replace=False, p=wt / wt.sum()):
            leave_now(i, m, pd.Timestamp("2025-03-31"), "Redundancy")

    # notice periods ending this month
    for i in np.where((status[:n] == 2) & (end_m[:n] == m))[0]:
        td = resign_date[i] + pd.DateOffset(months=int(end_m[i] - (resign_date[i].year - 2023) * 12 - resign_date[i].month + 1))
        leave_now(i, m, td, term_type[i])
        replace_manager(i, m)

    # December: performance reviews (hired by October)
    if moy == 11:
        for i in np.where(np.isin(status[:n], (1, 2)) & (hire_m[:n] <= m - 2))[0]:
            z = perf[i] * 0.8 + rng.normal(0, 0.6)
            rating[i] = 1 + np.searchsorted([-1.645, -0.842, 0.524, 1.405], z)
            reviews.append({"employee_id": eid(i), "review_year": year, "rating": int(rating[i]),
                            "reviewer_id": eid(mgr_of(i)) if mgr_of(i) >= 0 else ""})

    # new teams when teams grow past 12
    for d in (range(ND) if m < M - 1 else []):
        tl = [t for t in range(len(teams_dept)) if teams_dept[t] == d and not teams_lead[t]]
        ics = np.sum((dept[:n] == d) & np.isin(status[:n], (1, 2)) & (level[:n] <= 3))
        if ics / len(tl) > 12:
            seniors = np.where((dept[:n] == d) & (status[:n] == 1) & (level[:n] == 3))[0]
            if len(seniors):
                new = seniors[np.argmax(np.nan_to_num(rating[seniors], nan=3) + rng.random(len(seniors)))]
                level[new] = 4
                salary[new] = round(max(salary[new] * 1.12, band_mid(d, 4, loc[new], m + 1) * 0.95), -1)
                promo_m[new] = m + 1
                team[new] = lead_team[d]
                bad[new] = rng.random() < 0.10
                if bad[new]:
                    bad_ever.add(int(new))
                teams_dept.append(d); teams_mgr.append(new); teams_lead.append(False)
                log(new, mstart(m + 1), "Promotion")

    # hiring: open vacancies are filled at ~35 % a month (4 % during the freeze); new starters join next month
    if m < M - 1:
        fill = 0.04 if m in FREEZE else 0.35
        active1 = np.bincount(dept[np.where(status[:n] == 1)[0]], minlength=ND)
        for d in range(ND):
            need = int(max(0, round(budget[d] - active1[d])))
            for _ in range(rng.binomial(need, fill)):
                lvl = rng.choice([1, 2, 3], p=[0.45, 0.38, 0.17])
                pool = [j for j in term_type if term_type[j] == "Resignation" and status[j] == 3 and dept[j] == d
                        and j in term_date and (mstart(m) - term_date[j]).days > 180 and (rating[j] >= 3)
                        and j not in prev_id.values() and j not in [b[0] for b in boomerangs]]
                like = int(rng.choice(pool)) if pool and rng.random() < 0.05 else None
                i = new_person(d, lvl if like is None else max(lvl, int(level[like])), m + 1, like=like)
                if like is not None:
                    boomerangs.append((like, i))
                    if rng.random() < 0.4:
                        prev_id[i] = like
                hire_date[i] = rand_day(m + 1)
                promo_m[i] = m + 1
                set_team_for_new(i)
                log(i, hire_date[i], "Hire")

# ---------------------------------------------------------------- outputs
END = pd.Timestamp("2025-12-31")
emp_rows = []
for i in range(n):
    if hire_date[i] > END:
        assert status[i] == 1 and level[i] <= 3, 'only ordinary new starters may fall after the period'
        continue
    st = {1: "Active", 2: "Active", 3: "Terminated"}[status[i]]
    emp_rows.append({"employee_id": eid(i), "first_name": FIRST[fname[i]], "last_name": LAST[lname[i]],
                     "date_of_birth": pd.Timestamp(year=int(birth[i]), month=int(rng.integers(1, 13)), day=int(rng.integers(1, 29))).date(),
                     "gender": ["Female", "Male", "Prefer not to say"][gender[i]], "hire_date": hire_date[i].date(),
                     "department": DN[dept[i]], "job_title": title(i), "level": int(level[i]), "location": LOCS[loc[i]],
                     "work_mode": MODES[mode[i]], "manager_id": eid(mgr_of(i)) if 0 <= mgr_of(i) != i else "",
                     "fte": 0.8 if part_time[i] else 1.0, "annual_salary": round(float(salary[i]), 0),
                     "employment_status": st,
                     "resignation_date": resign_date[i].date() if i in resign_date else "",
                     "termination_date": term_date[i].date() if i in term_date else "",
                     "termination_type": term_type[i] if i in term_date else "",
                     "previous_employee_id": eid(prev_id[i]) if i in prev_id else ""})
E = pd.DataFrame(emp_rows)
# boomerang birth dates must match: copy the original's date of birth
dob = dict(zip(E.employee_id, E.date_of_birth))
for a, b in boomerangs:
    if eid(b) in dob:
        E.loc[E.employee_id == eid(b), "date_of_birth"] = dob[eid(a)]

H = pd.DataFrame(events)
H = H[H.effective_date <= END]
for c in ("effective_date", "hire_date", "last_promotion_date"):
    H[c] = pd.to_datetime(H[c]).dt.date
H = H.sort_values(["effective_date", "employee_id"], kind="stable").reset_index(drop=True)
TS = pd.DataFrame(timesheets)
AB = pd.DataFrame(absence)
SV = pd.DataFrame(survey)
SV.insert(0, "response_id", [f"R{100001 + k}" for k in range(len(SV))])
RV = pd.DataFrame(reviews)

# exit interviews: stated reason depends on the true main driver (people soften the real reason)
STATED = {
    "overtime_high": (["Workload / burnout", "Better opportunity", "Pay", "Personal / relocation"], [0.45, 0.35, 0.10, 0.10]),
    "below_band": (["Pay", "Better opportunity", "Career progression"], [0.70, 0.25, 0.05]),
    "promotion_stalled": (["Career progression", "Better opportunity", "Pay"], [0.65, 0.30, 0.05]),
    "low_engagement": (["Better opportunity", "Career progression", "Personal / relocation", "Workload / burnout", "Manager relationship"], [0.40, 0.20, 0.20, 0.10, 0.10]),
    "bad_manager": (["Manager relationship", "Better opportunity", "Career progression", "Personal / relocation"], [0.25, 0.45, 0.20, 0.10]),
    "rto_mandate": (["Flexibility / remote work", "Better opportunity"], [0.80, 0.20]),
    "first_year": (["Role not as expected", "Better opportunity", "Personal / relocation"], [0.50, 0.30, 0.20]),
    "none": (["Personal / relocation", "Better opportunity", "Further study"], [0.45, 0.40, 0.15]),
}
COMMENTS = {
    "Workload / burnout": ["The workload has been unsustainable for months.", "Too many late nights covering open roles.", "I need a better work-life balance."],
    "Better opportunity": ["I was offered a role I could not turn down.", "A great opportunity came up elsewhere.", "Moving to a bigger role at another company."],
    "Pay": ["The new offer is significantly higher.", "My salary has fallen behind the market.", "Pay did not keep up with my responsibilities."],
    "Career progression": ["I could not see a path to the next level.", "Promotion kept being pushed back.", "Looking for more growth."],
    "Manager relationship": ["I did not get the support I needed from my manager.", "Communication with my line manager was difficult."],
    "Flexibility / remote work": ["The office policy no longer works for me.", "I want to keep working remotely.", "The commute after the office change was too much."],
    "Role not as expected": ["The job was different from what was described.", "Not the right fit for me."],
    "Personal / relocation": ["Moving to another city.", "Family reasons.", "Personal circumstances changed."],
    "Further study": ["Going back to university.", "Starting a full-time course."],
}
xi, stated_vs_true = [], {}
for i, td in term_date.items():
    if term_type[i] != "Resignation" or td > END or rng.random() > 0.70:
        continue
    tr = true_driver.get(i, "none")
    opts, ps = STATED[tr]
    s = str(rng.choice(opts, p=ps))
    stated_vs_true.setdefault(tr, {}).setdefault(s, 0)
    stated_vs_true[tr][s] += 1
    xi.append({"employee_id": eid(i), "interview_date": (td - pd.Timedelta(days=int(rng.integers(1, 10)))).date(),
               "primary_reason": s, "would_return": rng.choice(["Yes", "Maybe", "No"], p=[0.35, 0.35, 0.30]),
               "comment": str(rng.choice(COMMENTS[s]))})
XI = pd.DataFrame(xi)

# salary bands
band_rows = []
for eff_m, eff in [(-9, "2022-04-01"), (3, "2023-04-01"), (15, "2024-04-01"), (27, "2025-04-01")]:
    for d in range(ND):
        for lv in range(1, 6):
            for lc in range(5):
                mid = band_mid(d, lv, lc, max(eff_m, 0) if eff_m >= 0 else -1)
                band_rows.append({"department": DN[d], "level": lv, "location": LOCS[lc], "effective_from": eff,
                                  "band_min": round(mid * 0.8, -2), "band_mid": round(mid, -2), "band_max": round(mid * 1.2, -2)})
BANDS = pd.DataFrame(band_rows)

# ---------------------------------------------------------------- truth (before the mess)
H_ = H.copy()
ytruth = {}
for y in (2023, 2024, 2025):
    hcs = []
    for mm in range(12):
        d0 = mstart((y - 2023) * 12 + mm)
        alive = E[(pd.to_datetime(E.hire_date) <= d0) & ((E.termination_date == "") | (pd.to_datetime(E.termination_date.replace("", None)) >= d0))]
        hcs.append(len(alive))
    hc_y = np.mean(hcs)
    t = E[(E.termination_date != "")]
    t = t[pd.to_datetime(t.termination_date).dt.year == y]
    ytruth[y] = {"avg_headcount": round(float(hc_y), 1), "leavers": int(len(t)),
                 "resignations": int((t.termination_type == "Resignation").sum()),
                 "voluntary_attrition_rate": round(float((t.termination_type == "Resignation").sum() / hc_y), 4),
                 "total_attrition_rate": round(float(len(t) / hc_y), 4),
                 "by_type": t.termination_type.value_counts().to_dict()}
tot_attr = sum(acc_attr.values())
truth = {
    "company": "Orrin Digital Ltd (fictional UK software and IT services company)",
    "period": "Jan 2023 - Dec 2025 (monthly simulation)",
    "hazard_model": {"form": "monthly logit P(resign) for staff not already on notice",
                     "intercept": A0, "coefficients": COEF,
                     "odds_ratios": {k: round(float(np.exp(v)), 2) for k, v in COEF.items()},
                     "department_shift": dict(zip(DN, D_SHIFT.tolist())), "month_of_year_shift": SEASON.tolist(),
                     "market_shift": MARKET,
                     "flag_definitions": {
                         "overtime_high": "average overtime over the last 3 months > 20 h a month",
                         "below_band": "salary / band midpoint (dept, level, location, current band) < 0.90",
                         "promotion_stalled": "level 1-3 and 36+ months since last promotion or hire",
                         "low_engagement": "true engagement < 2.9 (survey observes it with noise; disengaged staff respond less)",
                         "bad_manager": "line manager is one of the generator's 'bad managers' (not directly in the data)",
                         "rto_mandate": "was fully remote at an office location and moved to hybrid on 1 Sep 2024",
                         "first_year": "tenure < 12 months",
                         "long_tenure": "tenure >= 96 months"}},
    "attributable_resignations": {
        "note": "expected resignations that would not have happened if only this flag were switched off (direct effect)",
        "expected_total": round(float(acc_expected), 1),
        **{k: round(float(v), 1) for k, v in sorted(acc_attr.items(), key=lambda kv: -kv[1])}},
    "top_three_drivers": [k for k, _ in sorted(acc_attr.items(), key=lambda kv: -kv[1])[:3]],
    "engagement_model": "engagement = latent - 0.45 overtime_high - 0.40 bad_manager - 0.30 below_band - 0.35 rto_mandate - 0.20 promotion_stalled + noise",
    "driver_prevalence_by_month": prev_rows,
    "department_month": dept_month,
    "events": {"rto_mandate": "2024-09-01", "hiring_freeze": "2025-01 to 2025-06 (fill rate 4 % a month instead of 35 %)",
               "redundancy": {"date": "2025-03-31", "department": "Sales & Marketing", "people": REDUNDANCY_N},
               "pay_review": PAY_RISE, "band_uplift": APRILS},
    "by_year": ytruth,
    "bad_managers_ever": sorted(eid(i) for i in bad_ever),
    "boomerangs": [{"original": eid(a), "rehire": eid(b), "linked_in_hris": b in prev_id} for a, b in boomerangs],
    "exit_interview_stated_vs_true": stated_vs_true,
}

# ---------------------------------------------------------------- the mess a real HRIS export has
mess = {}
# 1. department name variants
VAR = {"Engineering": ["Eng", "engineering", "Engineering "], "Data & AI": ["Data and AI", "Data&AI"],
       "Product & Design": ["Product and Design"], "Delivery": ["Delivery / PMO"], "Customer Support": ["Support", "Cust. Support"],
       "Sales & Marketing": ["Sales and Marketing", "S&M"], "Corporate Services": ["Corporate", "Corp Services"]}
for df, name in ((E, "employees"), (H, "job_history")):
    idx = df.sample(frac=0.04, random_state=1).index
    df.loc[idx, "department"] = [rng.choice(VAR[d]) for d in df.loc[idx, "department"]]
    mess[f"{name}_department_variants"] = int(len(idx))
# 2. mixed date formats in employees.hire_date (dd/mm/yyyy)
idx = E.sample(frac=0.15, random_state=2).index
E["hire_date"] = E.hire_date.astype(str)
E.loc[idx, "hire_date"] = pd.to_datetime(E.loc[idx, "hire_date"]).dt.strftime("%d/%m/%Y")
mess["hire_date_ddmmyyyy"] = int(len(idx))
# 3. salary keyed as a monthly figure
idx = E.sample(frac=0.02, random_state=3).index
E.loc[idx, "annual_salary"] = (E.loc[idx, "annual_salary"] / 12).round(0)
mess["salary_entered_monthly"] = int(len(idx))
# 4. gender coding
g = E.gender.copy()
idx = E.sample(frac=0.10, random_state=4).index
E.loc[idx, "gender"] = [{"Female": rng.choice(["F", "female"]), "Male": rng.choice(["M", "male"]), "Prefer not to say": "PNTS"}[x] for x in g[idx]]
mess["gender_codes"] = int(len(idx))
# 5. stale status: terminated people still "Active"
idx = E[E.employment_status == "Terminated"].sample(9, random_state=5).index
E.loc[idx, "employment_status"] = "Active"
mess["terminated_but_status_active"] = sorted(E.loc[idx, "employee_id"].tolist())
# 6. missing manager ids
idx = E[E.manager_id != ""].sample(frac=0.01, random_state=6).index
E.loc[idx, "manager_id"] = ""
mess["manager_id_blanked"] = int(len(idx))
# 7. test records
tests = pd.DataFrame([{"employee_id": f"TEST{k}", "first_name": "Test", "last_name": nm, "date_of_birth": "1900-01-01",
                       "gender": "", "hire_date": "2024-01-01", "department": "Engineering", "job_title": "Software Engineer",
                       "level": 2, "location": "London", "work_mode": "Office", "manager_id": "", "fte": 1.0,
                       "annual_salary": 0, "employment_status": "Active", "resignation_date": "", "termination_date": "",
                       "termination_type": "", "previous_employee_id": ""} for k, nm in enumerate(["User", "DO NOT USE", "Payroll check", "User 2", "Integration", "Sandbox"], 1)])
E = pd.concat([E, tests], ignore_index=True).sample(frac=1, random_state=7).reset_index(drop=True)
mess["test_records"] = len(tests)
# 8. duplicate rows in job history (double-posted events)
dup = H.sample(18, random_state=8)
H = pd.concat([H, dup]).sort_values(["effective_date", "employee_id"], kind="stable").reset_index(drop=True)
mess["job_history_duplicate_rows"] = 18
# 9. timesheets: missing submissions and negative corrections
drop = TS.sample(frac=0.04, random_state=9).index
TS = TS.drop(drop).reset_index(drop=True)
mess["timesheets_missing"] = int(len(drop))
idx = TS.sample(frac=0.002, random_state=10).index
TS.loc[idx, "overtime_hours"] = -TS.loc[idx, "overtime_hours"].clip(lower=2)
mess["timesheets_negative_overtime"] = int(len(idx))
truth["injected_mess"] = mess
truth["survey_response_rate_low_vs_rest"] = np.mean(survey_low_resp, axis=0).round(3).tolist()
truth["survey_response_rate"] = round(len(SV) / survey_invited, 3)

E.to_csv(RAW / "employees.csv", index=False)
H.to_csv(RAW / "job_history.csv", index=False)
BANDS.to_csv(RAW / "salary_bands.csv", index=False)
RV.to_csv(RAW / "performance_reviews.csv", index=False)
SV.to_csv(RAW / "engagement_survey.csv", index=False)
TS.to_csv(RAW / "timesheets.csv", index=False)
AB.to_csv(RAW / "absence.csv", index=False)
XI.to_csv(RAW / "exit_interviews.csv", index=False)
pd.DataFrame(budget_rows).to_csv(RAW / "headcount_budget.csv", index=False)
(RAW / "true_effects.json").write_text(json.dumps(truth, indent=2, default=str))

print(f"people ever employed: {len(E) - len(tests)}, job history rows: {len(H)}, timesheet rows: {len(TS)}, survey rows: {len(SV)}")
for y, v in ytruth.items():
    print(y, v)
print("attributable:", truth["attributable_resignations"])
print("top three:", truth["top_three_drivers"])
pv = pd.DataFrame(prev_rows).set_index("month")
print(pv.iloc[[0, 11, 20, 26, 29, 35]].round(3))
dm = pd.DataFrame(dept_month)
print(dm.pivot(index="month", columns="department", values="gap").iloc[[0, 6, 12, 23, 26, 29, 32, 35]].round(3).to_string())
print(dm.pivot(index="month", columns="department", values="overtime_high_share").iloc[[0, 12, 23, 26, 29, 35]].to_string())
print("bad managers ever:", len(bad_ever), "boomerangs:", len(boomerangs))

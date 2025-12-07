"""
PATH WEEKEND OPTIMIZATION — Gurobi MILP (linearized + PWL for 1/f)

Saves a CSV summary to /mnt/data/path_schedule_solution_summary.csv

Author: adapted by ChatGPT
"""

import math
import re
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# -------------------------
# USER CONFIG
# -------------------------
DATA_DIR = Path("./data")
SAT_FILE = DATA_DIR / "Predicted_demand_saturday.xlsx"
SUN_FILE = DATA_DIR / "Predicted_demand_sunday.xlsx"

# Lines and parameters (edit to match your system)
LINES = ["NWK−WTC", "JSQ−33st", "HOB−WTC"]

T_cycle = {"NWK−WTC": 1.5, "JSQ−33st": 2.0, "HOB−WTC": 1.2}  # hours per round trip
K_car = 110  # seats per car
Y_max = 422  # fleet cars available
C_cost = 154.0  # $ per car-hour
Budget = 1e9

# Headway bounds (hours)
H_min = 3.0 / 60.0
H_max = 20.0 / 60.0

# Cars per train options
Len_min = 4
Len_max = 10
LEN_OPTIONS = list(range(Len_min, Len_max + 1))

# Time period definitions
TIME_PERIODS = ["AM", "Midday", "PM", "Evening"]
PERIOD_HOURS = {
    "AM": list(range(6, 10)),        # 06-09
    "Midday": list(range(10, 16)),   # 10-15
    "PM": list(range(16, 20)),       # 16-19
}
_all = set(range(24))
_assigned = set(sum([PERIOD_HOURS[p] for p in PERIOD_HOURS], []))
PERIOD_HOURS["Evening"] = sorted(list(_all - _assigned))
D_ur_t = {p: float(len(PERIOD_HOURS[p])) for p in TIME_PERIODS}

# Objective weights
w_time = 20.0
w_empty = 0.5

# Optional mapping file
OD_TO_LINE_CSV = DATA_DIR / "OD_to_line.csv"
stop_to_line = {}

# -------------------------
# EXCEL / OD LOADER (same robust heuristics)
# -------------------------
def normalize_sheet_name_to_hour(name):
    s = str(name).strip()
    m = re.match(r'^([01]?\d|2[0-3])(?:00)?$', s)
    if m:
        return int(m.group(1))
    digits = re.findall(r'\d+', s)
    if digits:
        val = int(digits[0])
        if 0 <= val <= 23:
            return val
    return None

def is_string_like(x):
    return isinstance(x, str) and x.strip() != ""

def find_matrix_origin(sheet_df, max_search_rows=12, max_search_cols=12):
    for r in range(min(max_search_rows, sheet_df.shape[0])):
        for c in range(min(max_search_cols, sheet_df.shape[1])):
            row_segment = sheet_df.iloc[r, c:c+8].tolist()
            col_segment = sheet_df.iloc[r:r+8, c].tolist()
            if sum(is_string_like(v) for v in row_segment) >= 2 and sum(is_string_like(v) for v in col_segment) >= 2:
                return r, c
    return None

def read_od_from_sheet(path: Path, sheet_name):
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, dtype=object)
    if raw.isna().all().all():
        return pd.DataFrame()
    origin = find_matrix_origin(raw)
    if origin is None:
        # fallback: try header=0 with index_col=0
        try:
            df_try = pd.read_excel(path, sheet_name=sheet_name, index_col=0)
            df_try.index = df_try.index.astype(str)
            df_try.columns = df_try.columns.astype(str)
            df_try = df_try.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            return df_try
        except Exception:
            return pd.DataFrame()
    r0, c0 = origin
    dests = []
    c = c0 + 1
    while c < raw.shape[1]:
        val = raw.iat[r0, c]
        if pd.isna(val):
            break
        dests.append(str(val))
        c += 1
    origins = []
    r = r0 + 1
    while r < raw.shape[0]:
        val = raw.iat[r, c0]
        if pd.isna(val):
            break
        origins.append(str(val))
        r += 1
    if len(origins) == 0 or len(dests) == 0:
        return pd.DataFrame()
    block = raw.iloc[r0+1:r0+1+len(origins), c0+1:c0+1+len(dests)].map(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0.0)
    block.index = origins
    block.columns = dests
    return block

def load_hourly(path: Path):
    xls = pd.ExcelFile(path)
    hour_od = {}
    for s in xls.sheet_names:
        h = normalize_sheet_name_to_hour(s)
        if h is None:
            continue
        hour_od[h] = read_od_from_sheet(path, s)
    for h in range(24):
        hour_od.setdefault(h, pd.DataFrame())
    return hour_od

def aggregate_hours_to_periods(hour_od_maps):
    period_od = {}
    for period, hours in PERIOD_HOURS.items():
        origins = set()
        dests = set()
        for h in hours:
            df = hour_od_maps.get(h, pd.DataFrame())
            if df is None or df.empty:
                continue
            origins.update(df.index.tolist())
            dests.update(df.columns.tolist())
        if len(origins) == 0 or len(dests) == 0:
            period_od[period] = pd.DataFrame()
            continue
        origins = sorted(origins); dests = sorted(dests)
        agg = pd.DataFrame(0.0, index=origins, columns=dests)
        for h in hours:
            df = hour_od_maps.get(h, pd.DataFrame())
            if df is None or df.empty:
                continue
            df_num = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            for o in df_num.index:
                for d in df_num.columns:
                    agg.at[o, d] += float(df_num.at[o, d])
        period_od[period] = agg
    return period_od

def load_od_to_line(path: Path):
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    mapping = {}
    for _, row in df.iterrows():
        o = str(row["origin"]); d = str(row["destination"])
        raw = row.get("line_list", "")
        if pd.isna(raw): continue
        lines = [s.strip() for s in str(raw).split(",") if s.strip()]
        mapping[(o, d)] = lines
    return mapping

def aggregate_od_to_line_demands(od_df, od_to_line):
    demand_by_line = {l: 0.0 for l in LINES}
    for o in od_df.index:
        for d in od_df.columns:
            q = float(od_df.at[o, d])
            if q <= 0: continue
            key = (o, d)
            lines = od_to_line.get(key, None)
            if lines is None:
                lo = stop_to_line.get(o); ld = stop_to_line.get(d)
                if lo and lo == ld:
                    lines = [lo]
                elif lo:
                    lines = [lo]
                elif ld:
                    lines = [ld]
                else:
                    lines = LINES
            split = q / len(lines)
            for L in lines:
                demand_by_line[L] += split
    return demand_by_line

# -------------------------
# MODEL BUILDING (linearized)
# -------------------------
def build_and_solve(period_demands, day_name="Saturday", pwl_points=8, verbose=True):
    od_to_line = load_od_to_line(OD_TO_LINE_CSV)
    demand_lt = {t: aggregate_od_to_line_demands(period_demands[t], od_to_line) for t in TIME_PERIODS}

    m = gp.Model(f"PATH_{day_name}")
    if not verbose:
        m.setParam("OutputFlag", 0)

    # compute frequency bounds and N_max for each line (used in big-M)
    f_lo = 1.0 / H_max
    f_hi = 1.0 / H_min
    N_max_by_line = {L: int(math.ceil(f_hi * T_cycle[L])) for L in LINES}

    # Variables
    N = {}               # integer number of trains (concurrent) in period (or per cycle)
    f = {}               # frequency (trains/hour)
    # We'll represent C by binaries z_{k} (one-hot) per (L,t)
    z = {}               # binary selection for car length k
    Y = {}               # Y_{k} = N * z_k  (integer or continuous) used to linearize N*C
    E = {}               # empty seats per hour (nonnegative)
    W = {}               # w = 1/f (PWL variable) to compute waiting time

    for L in LINES:
        for t in TIME_PERIODS:
            name = f"{L}_{t}"
            N[L, t] = m.addVar(vtype=GRB.INTEGER, name=f"N_{name}", lb=0, ub=N_max_by_line[L])
            f[L, t] = m.addVar(vtype=GRB.CONTINUOUS, name=f"f_{name}", lb=f_lo, ub=f_hi)
            # one-hot binaries for length
            for k in LEN_OPTIONS:
                z[L, t, k] = m.addVar(vtype=GRB.BINARY, name=f"z_{name}_{k}")
            # product vars Y_k = N * z_k (continuous, domain 0..N_max)
            for k in LEN_OPTIONS:
                Y[L, t, k] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=N_max_by_line[L], name=f"Y_{name}_{k}")
            E[L, t] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"E_{name}")
            W[L, t] = m.addVar(vtype=GRB.CONTINUOUS, lb=1.0 / f_hi, ub=1.0 / f_lo, name=f"W_{name}")  # w approx 1/f

    # Constraints
    for L in LINES:
        for t in TIME_PERIODS:
            # link f and N: f = N / T_cycle
            m.addConstr(f[L, t] == N[L, t] / T_cycle[L], name=f"link_f_N_{L}_{t}")

            # enforce exactly one length choice
            m.addConstr(gp.quicksum(z[L, t, k] for k in LEN_OPTIONS) == 1, name=f"onehot_{L}_{t}")

            # link Y_k = N * z_k using big-M (M = N_max_by_line[L])
            M = N_max_by_line[L]
            for k in LEN_OPTIONS:
                # Y <= M * z
                m.addConstr(Y[L, t, k] <= M * z[L, t, k], name=f"Y_ub1_{L}_{t}_{k}")
                # Y <= N
                m.addConstr(Y[L, t, k] <= N[L, t], name=f"Y_ub2_{L}_{t}_{k}")
                # Y >= N - M*(1 - z)
                m.addConstr(Y[L, t, k] >= N[L, t] - M * (1 - z[L, t, k]), name=f"Y_lb_{L}_{t}_{k}")
                # Y >= 0 (implicit)

            # seats per hour (linear): seats_ph = (sum_k k * Y_k) * (K_car / T_cycle[L])
            # demand per hour:
            dem = demand_lt[t][L]
            demand_ph = dem / max(1e-9, D_ur_t[t])
            seats_coeff = K_car / T_cycle[L]
            seats_ph_expr = gp.quicksum(k * Y[L, t, k] for k in LEN_OPTIONS) * seats_coeff

            # capacity constraint over period: seats_per_hour * hours_in_period >= demand (over period)
            m.addConstr(seats_ph_expr * D_ur_t[t] >= dem, name=f"cap_{L}_{t}")

            # empty seats per hour: E >= seats_ph - demand_ph
            m.addConstr(E[L, t] >= seats_ph_expr - demand_ph, name=f"empty_def_{L}_{t}")
            m.addConstr(E[L, t] >= 0, name=f"empty_nonneg_{L}_{t}")

            # PWL: W = 1/f => W(f) = 1/f. Use breakpoints between f_lo and f_hi
            # create breakpoints (avoid f very close to zero, bounded by f_lo/f_hi)
            xs = np.linspace(f_lo, f_hi, pwl_points)
            ys = [1.0 / x for x in xs]
            # add PWL constraint y = PWL(x)
            m.addGenConstrPWL(f[L, t], W[L, t], xs.tolist(), ys, name=f"pwl_inv_{L}_{t}")

    # Fleet constraint per period (cars concurrently used) sum_k (N * k) <= Y_max
    for t in TIME_PERIODS:
        fleet_expr = gp.quicksum( gp.quicksum(k * Y[L, t, k] for k in LEN_OPTIONS) for L in LINES )
        m.addConstr(fleet_expr <= Y_max, name=f"fleet_{t}")

    # Budget (operating cost) linear: total_car_hours = sum (N * C * D_ur_t) where N*C = sum_k k * Y_k
    total_car_hours = gp.quicksum( (gp.quicksum(k * Y[L, t, k] for k in LEN_OPTIONS)) * D_ur_t[t] for L in LINES for t in TIME_PERIODS )
    m.addConstr(C_cost * total_car_hours <= Budget, name="budget")

    # Objective: waiting + empty penalty + operating cost
    wait_terms = []
    empty_terms = []
    for t in TIME_PERIODS:
        for L in LINES:
            dem = demand_lt[t][L]
            # waiting hours = dem * (1/(2f)) = dem * 0.5 * W
            wait_terms.append(w_time * dem * 0.5 * W[L, t])
            # empty penalty over period = w_empty * E * hours_in_period
            empty_terms.append(w_empty * E[L, t] * D_ur_t[t])

    obj = gp.quicksum(wait_terms) + gp.quicksum(empty_terms) + C_cost * total_car_hours
    m.setObjective(obj, GRB.MINIMIZE)

    # Solve
    m.Params.TimeLimit = 600  # seconds, optional
    m.optimize()

    # Extract solution
    sol = {"objective": m.ObjVal}
    sol["N"] = {(L, t): int(round(N[L, t].X)) for L in LINES for t in TIME_PERIODS}
    # recover chosen car length per (L,t)
    sol["cars"] = {}
    for L in LINES:
        for t in TIME_PERIODS:
            chosen_k = None
            for k in LEN_OPTIONS:
                if z[L, t, k].X > 0.5:
                    chosen_k = k
                    break
            sol["cars"][(L, t)] = chosen_k if chosen_k is not None else LEN_OPTIONS[0]
    # frequency from f var
    sol["freq"] = {(L, t): float(f[L, t].X) for L in LINES for t in TIME_PERIODS}

    return sol

# -------------------------
# RUN & SAVE
# -------------------------
def main():
    print("Loading hourly OD for Saturday...")
    sat_hours = load_hourly(SAT_FILE)
    sat_periods = aggregate_hours_to_periods(sat_hours)

    print("Loading hourly OD for Sunday...")
    sun_hours = load_hourly(SUN_FILE)
    sun_periods = aggregate_hours_to_periods(sun_hours)

    print("Solving Saturday...")
    sat_sol = build_and_solve(sat_periods, day_name="Saturday", pwl_points=8, verbose=True)
    print("Solving Sunday...")
    sun_sol = build_and_solve(sun_periods, day_name="Sunday", pwl_points=8, verbose=True)

    # Summarize & save
    rows = []
    for day_name, sol in [("Saturday", sat_sol), ("Sunday", sun_sol)]:
        print(f"\n{day_name} objective: {sol['objective']:.2f}")
        for t in TIME_PERIODS:
            print(f" Period {t}:")
            for L in LINES:
                print(f"  {L}: N={sol['N'][(L,t)]}, cars={sol['cars'][(L,t)]}, freq={sol['freq'][(L,t)]:.3f} tr/hr")
                rows.append({
                    "day": day_name, "period": t, "line": L,
                    "N": sol['N'][(L,t)], "cars_per_train": sol['cars'][(L,t)],
                    "freq_tr_per_hr": sol['freq'][(L,t)]
                })

    out = pd.DataFrame(rows)
    outpath = "./output/path_schedule_solution_summary.csv"
    out.to_csv(outpath, index=False)
    print(f"\nSaved summary to {outpath}")

        # --------------------------
    # PERFORMANCE METRICS
    # --------------------------
    rows_metrics = []

    def compute_metrics(periods_od, solution, day_name):
        for t in TIME_PERIODS:
            hours_in_period = D_ur_t[t]
            od_df = periods_od[t]

            # Compute demand per line
            od_to_line = load_od_to_line(OD_TO_LINE_CSV)
            demand_line = aggregate_od_to_line_demands(od_df, od_to_line)

            for L in LINES:
                dem_total = demand_line[L]                   # total passengers in period
                dem_per_hr = dem_total / hours_in_period    # passengers per hour

                freq = solution["freq"][(L, t)]             # trains/hr
                cars = solution["cars"][(L, t)]             # cars/train
                seats_per_train = cars * K_car              # seats/train
                seats_per_hour = seats_per_train * freq     # seats/hr

                # basic metrics
                headway_min = (60.0 / freq) if freq > 0 else None
                wait_min = (headway_min / 2.0) if headway_min else None
                utilization = (dem_per_hr / seats_per_hour * 100) if seats_per_hour > 0 else 0
                empty_hr = max(seats_per_hour - dem_per_hr, 0)

                rows_metrics.append({
                    "day": day_name,
                    "period": t,
                    "line": L,
                    "demand_total": dem_total,
                    "hours_in_period": hours_in_period,
                    "demand_per_hour": dem_per_hr,
                    "cars_per_train": cars,
                    "seats_per_train": seats_per_train,
                    "freq_trains_per_hr": freq,
                    "headway_min": headway_min,
                    "wait_time_min": wait_min,
                    "seats_per_hour": seats_per_hour,
                    "empty_seats_per_hour": empty_hr,
                    "utilization_percent": utilization
                })

    compute_metrics(sat_periods, sat_sol, "Saturday")
    compute_metrics(sun_periods, sun_sol, "Sunday")

    metrics_df = pd.DataFrame(rows_metrics)
    out_metrics = "./output/path_schedule_performance_metrics.csv"
    metrics_df.to_csv(out_metrics, index=False)
    print(f"Saved performance metrics to {out_metrics}")


if __name__ == "__main__":
    main()

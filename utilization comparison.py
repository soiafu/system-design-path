import pandas as pd
import numpy as np
from pathlib import Path
import re
import math

# -------------------------
# 1. Configuration & Parameters
# -------------------------
DATA_DIR = Path("./data")
SAT_FILE = DATA_DIR / "Predicted_demand_saturday.xlsx"
SUN_FILE = DATA_DIR / "Predicted_demand_sunday.xlsx"
OD_TO_LINE_CSV = DATA_DIR / "OD_to_line.csv"
SOL_SUMMARY_CSV = Path("./output/path_schedule_solution_summary.csv")
OUTPUT_CSV = Path("./output/path_utilization_comparison.csv")

# Line Definitions
LINES = ["NWK−WTC", "JSQ−33st", "HOB−33st"]

# Car Parameters
K_car = 110  # Seats per car

# Benchmark System Parameters
BENCH_FREQ = 3.0       # 3 trains per hour

# --- MODIFICATION: Dynamic Benchmark Car Lengths ---
def get_benchmark_cars(line_name):
    """
    Returns the benchmark train length based on the line.
    NWK-WTC uses 9-car trains; others use 8-car trains.
    """
    # If it is the NWK-WTC line, return 9 cars
    if "NWK" in line_name and "WTC" in line_name:
        return 9
    # For other lines (JSQ-33st, HOB-33st), return 8 cars
    return 8

# Time Period Definitions (used to calculate per-hour demand)
PERIOD_HOURS = {
    "AM": list(range(6, 10)),        # 4 hours
    "Midday": list(range(10, 16)),   # 6 hours
    "PM": list(range(16, 20)),       # 4 hours
    "Evening": []                    # Calculated below
}
_all = set(range(24))
_assigned = set(sum([PERIOD_HOURS[p] for p in ["AM", "Midday", "PM"]], []))
PERIOD_HOURS["Evening"] = sorted(list(_all - _assigned)) # Remaining 10 hours
D_ur_t = {p: float(len(PERIOD_HOURS[p])) for p in PERIOD_HOURS}

# -------------------------
# 2. Data Loading Functions (Reusing original logic)
# -------------------------

def normalize_sheet_name_to_hour(name):
    s = str(name).strip()
    m = re.match(r'^([01]?\d|2[0-3])(?:00)?$', s)
    if m: return int(m.group(1))
    return None

def read_od_from_sheet(path, sheet_name):
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, header=None)
        # Simplified reading logic. Replace with robust logic if headers are complex.
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        return df
    except:
        return pd.DataFrame()

def load_hourly_data(path):
    xls = pd.ExcelFile(path)
    hour_od = {}
    for s in xls.sheet_names:
        h = normalize_sheet_name_to_hour(s)
        if h is None: continue
        try:
            # Assuming standard format; adjust index_col if needed
            df = pd.read_excel(path, sheet_name=s, index_col=0)
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            hour_od[h] = df
        except:
            pass
    return hour_od

def aggregate_od_to_line(period_od_map, od_to_line_map):
    """Aggregates OD matrix demand to specific lines."""
    line_demand = {l: 0.0 for l in LINES}
    
    for df in period_od_map.values():
        if df.empty: continue
        for o in df.index:
            for d in df.columns:
                q = float(df.at[o, d])
                if q <= 0: continue
                
                # Find mapping
                key = (str(o), str(d))
                lines = None
                for k_map, v_lines in od_to_line_map.items():
                    if str(k_map[0]) == str(o) and str(k_map[1]) == str(d):
                        lines = v_lines
                        break
                
                if lines is None:
                    # Fallback: distribute evenly if no mapping found
                    lines = LINES 
                
                split_q = q / len(lines)
                for l in lines:
                    line_demand[l] += split_q
    return line_demand

def load_od_map_file(path):
    if not path.exists(): return {}
    df = pd.read_csv(path)
    mapping = {}
    for _, row in df.iterrows():
        o, d = str(row['origin']), str(row['destination'])
        l_str = str(row.get('line_list', ''))
        lines = [x.strip() for x in l_str.split(',') if x.strip()]
        mapping[(o, d)] = lines
    return mapping

# -------------------------
# 3. Core Calculation Logic
# -------------------------

def main():
    print("Loading Solution Summary...")
    if not SOL_SUMMARY_CSV.exists():
        print(f"Error: {SOL_SUMMARY_CSV} not found. Run optimization first.")
        return
    sol_df = pd.read_csv(SOL_SUMMARY_CSV)
    
    print("Loading OD Map...")
    od_map = load_od_map_file(OD_TO_LINE_CSV)
    
    results = []

    days_config = [
        ("Saturday", SAT_FILE),
        ("Sunday", SUN_FILE)
    ]

    for day_name, file_path in days_config:
        print(f"Processing {day_name} demand from {file_path}...")
        
        # 1. Load hourly data
        hour_data = load_hourly_data(file_path) 
        
        # 2. Iterate through periods
        for period in PERIOD_HOURS:
            hours = PERIOD_HOURS[period]
            period_hourly_dfs = {h: hour_data.get(h, pd.DataFrame()) for h in hours}
            
            # 3. Aggregate total demand for the period
            line_total_demand = aggregate_od_to_line(period_hourly_dfs, od_map)
            
            hours_duration = D_ur_t[period]
            if hours_duration == 0: hours_duration = 1.0

            for line in LINES:
                total_dem = line_total_demand.get(line, 0.0)
                # Calculate average demand per hour (Pax/Hr)
                pax_per_hr = total_dem / hours_duration
                
                # --- MODIFICATION: Calculate Benchmark Capacity Dynamically ---
                # Determine cars based on line name (9 for NWK-WTC, 8 for others)
                current_bench_cars = get_benchmark_cars(line)
                
                # Bench Capacity = 3 trains/hr * [8 or 9] cars * 110 seats
                bench_cap_hr = BENCH_FREQ * current_bench_cars * K_car
                
                bench_util = (pax_per_hr / bench_cap_hr * 100) if bench_cap_hr > 0 else 0
                
                # --- Get Optimized Metrics ---
                row = sol_df[
                    (sol_df['day'] == day_name) & 
                    (sol_df['period'] == period) & 
                    (sol_df['line'] == line)
                ]
                
                if not row.empty:
                    opt_freq = float(row.iloc[0]['freq_tr_per_hr'])
                    opt_cars = float(row.iloc[0]['cars_per_train'])
                    opt_cap_hr = opt_freq * opt_cars * K_car
                    opt_util = (pax_per_hr / opt_cap_hr * 100) if opt_cap_hr > 0 else 0
                else:
                    opt_freq = 0; opt_cars = 0; opt_cap_hr = 0; opt_util = 0

                results.append({
                    "Day": day_name,
                    "Period": period,
                    "Line": line,
                    "Demand_Pax_Hr": round(pax_per_hr, 1),
                    # Benchmark columns
                    "Bench_Freq": BENCH_FREQ,
                    "Bench_Cars": current_bench_cars,  # Logs the specific car length used
                    "Bench_Cap_Hr": int(bench_cap_hr),
                    "Bench_Util_Pct": round(bench_util, 2),
                    # Optimized columns
                    "Opt_Freq": round(opt_freq, 2),
                    "Opt_Cars": int(opt_cars),
                    "Opt_Cap_Hr": int(opt_cap_hr),
                    "Opt_Util_Pct": round(opt_util, 2),
                    # Comparison
                    "Util_Change_Pct": round(opt_util - bench_util, 2)
                })

    # 4. Save Results
    out_df = pd.DataFrame(results)
    
    print("\n--- Top 5 Highest Utilization Scenarios (Benchmark) ---")
    # Displaying the most crowded periods in the benchmark scenario
    print(out_df.sort_values("Bench_Util_Pct", ascending=False).head(5)[
        ["Day", "Period", "Line", "Bench_Cars", "Bench_Util_Pct", "Opt_Util_Pct"]
    ].to_string(index=False))

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved utilization comparison to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Load optimized metrics
# -------------------------------
metrics = pd.read_csv("./output/path_schedule_performance_metrics.csv")

# -------------------------------
# ENTER YOUR BENCHMARK HERE
# -------------------------------
# Example baseline headways in minutes (YOU MUST ADJUST THESE)
baseline_headways = {
    "NWK−WTC": {"AM": 10, "Midday": 10, "PM": 10, "Evening": 15},
    "JSQ−33st": {"AM": 8, "Midday": 8, "PM": 8, "Evening": 10},
    "HOB−WTC": {"AM": 10, "Midday": 10, "PM": 10, "Evening": 15},
}

# Compute baseline metrics
baseline_rows = []
for line in baseline_headways:
    for period in baseline_headways[line]:
        hw = baseline_headways[line][period]
        freq = 60.0 / hw
        wait = hw / 2.0
        baseline_rows.append({
            "line": line,
            "period": period,
            "baseline_headway": hw,
            "baseline_wait": wait,
            "baseline_freq": freq
        })

baseline = pd.DataFrame(baseline_rows)

# Merge with optimized
merged = metrics.merge(baseline, on=["line", "period"], how="left")

# -------------------------------
# Plot 1: Wait Time Comparison
# -------------------------------
plt.figure(figsize=(10,6))
for L in merged["line"].unique():
    sub = merged[merged["line"] == L]
    plt.plot(sub["period"], sub["baseline_wait"], label=f"{L} baseline", linestyle="--")
    plt.plot(sub["period"], sub["wait_time_min"], label=f"{L} optimized")
plt.title("Passenger Wait Time: Baseline vs Optimized")
plt.ylabel("Wait Time (min)")
plt.legend()
plt.grid(True)
plt.savefig("wait_time_comparison.png")
print("Saved wait_time_comparison.png")

# -------------------------------
# Plot 2: Headway Comparison
# -------------------------------
plt.figure(figsize=(10,6))
for L in merged["line"].unique():
    sub = merged[merged["line"] == L]
    plt.plot(sub["period"], sub["baseline_headway"], label=f"{L} baseline", linestyle="--")
    plt.plot(sub["period"], sub["headway_min"], label=f"{L} optimized")
plt.title("Headway: Baseline vs Optimized")
plt.ylabel("Headway (minutes)")
plt.legend()
plt.grid(True)
plt.savefig("headway_comparison.png")
print("Saved headway_comparison.png")

# -------------------------------
# Plot 3: Utilization Comparison
# -------------------------------
plt.figure(figsize=(10,6))
for L in merged["line"].unique():
    sub = merged[merged["line"] == L]
    plt.plot(sub["period"], sub["utilization_percent"], label=f"{L} optimized")
plt.title("Utilization (%) of Train Capacity")
plt.ylabel("Utilization (%)")
plt.legend()
plt.grid(True)
plt.savefig("utilization_comparison.png")
print("Saved utilization_comparison.png")

# -------------------------------
# Plot 4: Seats per Hour Comparison
# -------------------------------
plt.figure(figsize=(10,6))
for L in merged["line"].unique():
    sub = merged[merged["line"] == L]
    plt.plot(sub["period"], sub["seats_per_hour"], label=f"{L} optimized")
plt.title("Seats Provided per Hour (Capacity Supply)")
plt.ylabel("Seats per Hour")
plt.legend()
plt.grid(True)
plt.savefig("capacity_supply_comparison.png")
print("Saved capacity_supply_comparison.png")

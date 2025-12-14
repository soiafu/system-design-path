import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------------
# 1. Configuration and Data Loading
# -------------------------------
INPUT_FILE = "./output/path_utilization_comparison.csv"
OUTPUT_DIR = "charts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"Error: File {INPUT_FILE} not found.")
    exit()

# -------------------------------
# 2. Data Cleaning
# -------------------------------
df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)

col_benchmark = 'Bench_Util_Pct'
col_optimized = 'Opt_Util_Pct'
col_period = 'Period'
col_line = 'Line'
col_day = 'Day'

df[col_benchmark] = pd.to_numeric(df[col_benchmark], errors='coerce')
df[col_optimized] = pd.to_numeric(df[col_optimized], errors='coerce')
df = df.dropna(subset=[col_benchmark, col_optimized])

period_order = ["AM", "Midday", "PM", "Evening"]
df[col_period] = pd.Categorical(df[col_period], categories=period_order, ordered=True)

lines = df[col_line].unique()
days = df[col_day].unique()

# -------------------------------
# 3. COLOR FUNCTION (NEW)
# -------------------------------
def get_colors(line_name):
    line = line_name.upper()

    if "NWK" in line or "WTC" in line:
        # NWK–WTC → Red
        return '#ffcccc', '#cc0000'
    elif "JSQ" in line and "33" in line:
        # JSQ–33rd → Yellow
        return '#fde68a', '#f59e0b'
    else:
        # HOB–33rd → Blue
        return '#add8e6', '#4682b4'

# -------------------------------
# 4. UTILIZATION COMPARISON PLOTS
# -------------------------------
n_lines = len(lines)
n_days = len(days)

fig, axes = plt.subplots(
    n_lines, n_days,
    figsize=(5 * n_days + 2, 4 * n_lines),
    sharey=True
)

if n_lines == 1 and n_days == 1:
    axes = [[axes]]
elif n_lines == 1:
    axes = [axes]
elif n_days == 1:
    axes = [[ax] for ax in axes]

bar_width = 0.35
y_max = df[[col_benchmark, col_optimized]].max().max() * 1.20

print("Generating utilization charts...")

for i, line in enumerate(lines):
    for j, day in enumerate(days):
        ax = axes[i][j]

        data = df[(df[col_line] == line) & (df[col_day] == day)].sort_values(by=col_period)

        if data.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue

        x = range(len(data))
        c_bench, c_opt = get_colors(line)

        rects1 = ax.bar(
            [k - bar_width / 2 for k in x],
            data[col_benchmark],
            width=bar_width,
            label='Benchmark',
            color=c_bench
        )

        rects2 = ax.bar(
            [k + bar_width / 2 for k in x],
            data[col_optimized],
            width=bar_width,
            label='Optimized',
            color=c_opt
        )

        ax.set_title(f"{line} - {day}", fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(data[col_period])

        if j == 0:
            ax.set_ylabel("Utilization (%)")

        ax.set_ylim(0, y_max)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # Label bars
        def add_labels(rects):
            for r in rects:
                h = r.get_height()
                ax.annotate(
                    f'{h:.0f}%',
                    xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    fontweight='bold'
                )

        add_labels(rects1)
        add_labels(rects2)

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))

plt.suptitle("Comparison of Seat Utilization: Benchmark vs. Optimized", fontsize=16, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])

util_save = os.path.join(OUTPUT_DIR, "utilization_comparison_labeled.png")
plt.savefig(util_save, dpi=300)
plt.close()

print(f"Saved: {util_save}")

# -------------------------------
# 5. WAIT TIME COMPARISON PLOTS
# -------------------------------
BENCH_FREQ = 3.0

df["Bench_Wait_Min"] = 60.0 / (2.0 * BENCH_FREQ)
df["Opt_Wait_Min"] = 60.0 / (2.0 * df["Opt_Freq"])

df = df.dropna(subset=["Bench_Wait_Min", "Opt_Wait_Min"])
wait_ymax = max(df["Bench_Wait_Min"].max(), df["Opt_Wait_Min"].max()) * 1.25

fig, axes = plt.subplots(
    n_lines, n_days,
    figsize=(5 * n_days + 2, 4 * n_lines),
    sharey=True
)

if n_lines == 1 and n_days == 1:
    axes = [[axes]]
elif n_lines == 1:
    axes = [axes]
elif n_days == 1:
    axes = [[ax] for ax in axes]

print("Generating wait time charts...")

for i, line in enumerate(lines):
    for j, day in enumerate(days):
        ax = axes[i][j]

        data = df[(df[col_line] == line) & (df[col_day] == day)].sort_values(by=col_period)

        if data.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue

        x = range(len(data))
        c_bench, c_opt = get_colors(line)

        rects1 = ax.bar(
            [k - bar_width / 2 for k in x],
            data["Bench_Wait_Min"],
            width=bar_width,
            label="Benchmark",
            color=c_bench
        )

        rects2 = ax.bar(
            [k + bar_width / 2 for k in x],
            data["Opt_Wait_Min"],
            width=bar_width,
            label="Optimized",
            color=c_opt
        )

        ax.set_title(f"{line} - {day}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(data[col_period])

        if j == 0:
            ax.set_ylabel("Average Wait Time (min)")

        ax.set_ylim(0, wait_ymax)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        def add_wait_labels(rects):
            for r in rects:
                h = r.get_height()
                ax.annotate(
                    f"{h:.1f}",
                    xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold"
                )

        add_wait_labels(rects1)
        add_wait_labels(rects2)

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))

plt.suptitle(
    "Comparison of Average Passenger Wait Time: Benchmark vs. Optimized",
    fontsize=16,
    y=0.99
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

wait_save = os.path.join(OUTPUT_DIR, "wait_time_comparison_labeled.png")
plt.savefig(wait_save, dpi=300)
plt.close()

print(f"Saved: {wait_save}")

import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------------
# 1. Configuration and Data Loading
# -------------------------------
INPUT_FILE = "./output/path_utilization_comparison.csv"
OUTPUT_DIR = "figure"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read the data
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
# 3. Plotting Logic
# -------------------------------
n_lines = len(lines)
n_days = len(days)

fig, axes = plt.subplots(n_lines, n_days, figsize=(5 * n_days + 2, 4 * n_lines), sharey=True)

if n_lines == 1 and n_days == 1:
    axes = [[axes]]
elif n_lines == 1:
    axes = [axes]
elif n_days == 1:
    axes = [[ax] for ax in axes]

bar_width = 0.35
# Increase Y-axis limit slightly to fit labels
y_max = df[[col_benchmark, col_optimized]].max().max() * 1.20 

print("Generating charts with full labels...")

for i, line in enumerate(lines):
    for j, day in enumerate(days):
        ax = axes[i][j]
        
        data = df[(df[col_line] == line) & (df[col_day] == day)].sort_values(by=col_period)
        
        if data.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue

        x = range(len(data))
        
        # Color Logic
        if "NWK" in line or "WTC" in line:
            c_bench = '#ffcccc' # Light Red
            c_opt = '#cc0000'   # Dark Red
        else:
            c_bench = '#add8e6' # Light Blue
            c_opt = '#4682b4'   # Steel Blue
        
        # Plot Bars
        rects1 = ax.bar([k - bar_width/2 for k in x], data[col_benchmark], 
                        width=bar_width, label='Benchmark', color=c_bench)
        rects2 = ax.bar([k + bar_width/2 for k in x], data[col_optimized], 
                        width=bar_width, label='Optimized', color=c_opt)
        
        # Titles and Ticks
        ax.set_title(f"{line} - {day}", fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(data[col_period])
        
        if j == 0:
            ax.set_ylabel("Utilization (%)")
            
        ax.set_ylim(0, y_max)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # -------------------------------------------------------
        # NEW: Function to add labels to both sets of bars
        # -------------------------------------------------------
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.0f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # Vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Apply labels to Benchmark bars
        add_labels(rects1)
        # Apply labels to Optimized bars
        add_labels(rects2)
        # -------------------------------------------------------

# -------------------------------
# 4. Layout and Saving
# -------------------------------
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98), fontsize=11)

plt.suptitle("Comparison of Seat Utilization: Benchmark vs. Optimized", fontsize=16, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.95])

save_path = os.path.join(OUTPUT_DIR, "utilization_comparison_labeled.png")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Charts successfully generated and saved to: {save_path}")
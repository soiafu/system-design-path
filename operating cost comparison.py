import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_cost_comparison(bench_daily_cost, proposed_costs):
    """
    Generates a grouped bar chart comparing Benchmark vs Proposed costs.
    The budget line has been removed as per the user's request.
    """
    # 1. Prepare Data
    days = ['Saturday', 'Sunday']
    # Extract values from the proposed_costs dictionary (default to 0 if missing)
    prop_sat = proposed_costs.get('Saturday', 0) if proposed_costs is not None else 0
    prop_sun = proposed_costs.get('Sunday', 0) if proposed_costs is not None else 0
    
    total_bench = bench_daily_cost * 2
    total_prop = prop_sat + prop_sun

    # X-axis labels: Saturday, Sunday, and Total Weekend
    labels = ['Saturday', 'Sunday', 'Total Weekend']
    bench_values = [bench_daily_cost, bench_daily_cost, total_bench]
    prop_values = [prop_sat, prop_sun, total_prop]
    
    x = np.arange(len(labels))
    width = 0.35  # Width of the bars

    # 2. Create the Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotting the bars
    # Using Navy Blue for Benchmark and Deep Orange for Proposed for a professional look
    rects1 = ax.bar(x - width/2, bench_values, width, label='Benchmark System', color='#2c3e50', alpha=0.8)
    rects2 = ax.bar(x + width/2, prop_values, width, label='Proposed System', color='#d35400', alpha=0.8)

    # 3. Styling and Labels
    ax.set_ylabel('Operating Cost ($)', fontsize=12)
    ax.set_title('Weekend Operating Cost: Benchmark vs. Proposed System', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend()
    
    # Add horizontal gridlines for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)

    # 4. Function to add data labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'${height:,.0f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),  # 5 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', 
                        fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    
    # 5. Save and Show
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "cost_comparison_chart.png"
    plt.savefig(save_path, dpi=300)
    
    print(f"\n[Success] Chart saved to: {save_path.absolute()}")
    plt.show()

# -------------------------
# 1. GLOBAL CONFIGURATION
# -------------------------
C_COST = 154.0  
T_CYCLE = {
    "NWK−WTC": 1.5, 
    "JSQ−33st": 2.0, 
    "HOB−33st": 1.2
}

FILE_PATH = './output/path_schedule_performance_metrics.csv'

# -------------------------
# 2. BENCHMARK SYSTEM DATA
# -------------------------
# According to your input: 60 trips total per line over the 24h window
BENCHMARK_TRIPS = 60.0
BENCHMARK_CARS = {
    "NWK−WTC": 9,
    "JSQ−33st": 8,
    "HOB−33st": 8,
}

def calculate_benchmark_cost():
    """Calculates cost based on fixed 33 trips per line."""
    total_car_hours = 0
    for line in T_CYCLE:
        trips = BENCHMARK_TRIPS
        cars = BENCHMARK_CARS[line]
        cycle = T_CYCLE[line]
        
        # Total Car-Hours = Trips * Round Trip Time * Cars
        line_car_hours = trips * cycle * cars
        total_car_hours += line_car_hours
        
    return total_car_hours * C_COST

# -------------------------
# 3. PROPOSED SYSTEM DATA
# -------------------------
def calculate_proposed_filtered_cost(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return None

    df = pd.read_csv(path)
    
    # Filter for daytime hours (excluding Evening)
    valid_periods = ["AM", "Midday", "PM", "Evening"]
    df_filtered = df[df['period'].isin(valid_periods)].copy()

    # Calculation: (freq * T_cycle * cars * duration) * C_cost
    df_filtered['operating_cost'] = df_filtered.apply(
        lambda row: (row['freq_trains_per_hr'] * T_CYCLE[row['line']] * row['cars_per_train']) 
                    * row['hours_in_period'] * C_COST, 
        axis=1
    )
    
    return df_filtered.groupby('day')['operating_cost'].sum()

# -------------------------
# 4. EXECUTION & COMPARISON
# -------------------------
def main():
    print("="*65)
    print("    COST ANALYSIS: BENCHMARK (33 TRIPS) VS PROPOSED SYSTEM")
    print("="*65)

    # 1. Benchmark Cost
    bench_cost = calculate_benchmark_cost()
    print(f"{'Benchmark System (33 trips/line):':<40} ${bench_cost:,.2f}")

    # 2. Proposed Cost
    proposed_costs = calculate_proposed_filtered_cost(FILE_PATH)

    if proposed_costs is not None:
        for day, cost in proposed_costs.items():
            print("-" * 65)
            print(f"Proposed System ({day}, 6AM-7PM):")
            print(f"{'Operational Cost:':<40} ${cost:,.2f}")
            
            diff = bench_cost - cost
            percent = (diff / bench_cost) * 100
            
            if diff >= 0:
                print(f"{'Savings vs Benchmark:':<40} ${diff:,.2f} ({percent:.2f}%)")
            else:
                print(f"{'Additional Cost vs Benchmark:':<40} ${abs(diff):,.2f} ({abs(percent):.2f}%)")

    print("="*65)
    plot_cost_comparison(bench_cost, proposed_costs)

if __name__ == "__main__":
    main()
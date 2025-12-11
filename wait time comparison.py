import pandas as pd
import numpy as np

def generate_benchmark_comparison(solution_csv_path, output_path):
    # 1. Read the optimization results
    df = pd.read_csv(solution_csv_path)

    # 2. Define benchmark parameters
    # Mapping line names to benchmark values
    # Red = NWK-WTC, Yellow = JSQ-33st, Blue = HOB-33st
    # All are currently 3 trains/hr
    BENCH_FREQ = 3.0  
    BENCH_WAIT = 10.0 # minutes

    # 3. Calculate optimized metrics
    # Opt Wait (min) = (60 / freq) / 2
    df['Opt_Freq'] = df['freq_tr_per_hr']
    df['Opt_Wait_min'] = (60.0 / df['Opt_Freq']) / 2.0
    
    # 4. Insert benchmark columns
    df['Bench_Freq'] = BENCH_FREQ
    df['Bench_Wait_min'] = BENCH_WAIT

    # 5. Compute differences (improvements)
    # Frequency increase (higher is better)
    df['Freq_Diff'] = df['Opt_Freq'] - df['Bench_Freq']
    
    # Wait time saved (higher is better; i.e., Bench - Opt)
    df['Wait_Saved_min'] = df['Bench_Wait_min'] - df['Opt_Wait_min']
    
    # Saved percentage
    df['Wait_Saved_Pct'] = (df['Wait_Saved_min'] / df['Bench_Wait_min']) * 100

    # 6. Arrange columns for display order
    cols = [
        'day', 'period', 'line', 
        'Bench_Freq', 'Opt_Freq', 'Freq_Diff',
        'Bench_Wait_min', 'Opt_Wait_min', 'Wait_Saved_min', 'Wait_Saved_Pct'
    ]
    
    comparison_df = df[cols].round(2)

    # 7. Print preview
    print("\n--- BENCHMARK COMPARISON (Top 10 Rows) ---")
    print(comparison_df.head(10).to_string(index=False))
    
    # 8. Save file
    comparison_df.to_csv(output_path, index=False)
    print(f"\nSaved comparison report to: {output_path}")

# --- Execute ---
if __name__ == "__main__":
    # Ensure these paths match your actual file locations
    input_csv = "./output/path_schedule_solution_summary.csv" 
    output_csv = "./output/path_benchmark_comparison.csv"
    
    try:
        generate_benchmark_comparison(input_csv, output_csv)
    except FileNotFoundError:
        print(f"Error: Could not find {input_csv}. Please run the optimization first.")
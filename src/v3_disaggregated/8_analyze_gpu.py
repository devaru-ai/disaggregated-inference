import pandas as pd
import sys

def analyze_logs(csv_file):
    df = pd.read_csv(csv_file)
    
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    
    # Strip strings and convert to actual numbers so pandas can do math
    df['utilization.gpu [%]'] = df['utilization.gpu [%]'].astype(str).str.replace('%', '').astype(float)
    df['utilization.memory [%]'] = df['utilization.memory [%]'].astype(str).str.replace('%', '').astype(float)
    df['memory.used [MiB]'] = df['memory.used [MiB]'].astype(str).str.replace('MiB', '').astype(float)

    avg_gpu = df['utilization.gpu [%]'].mean()
    avg_mem = df['utilization.memory [%]'].mean()
    max_mem_used = df['memory.used [MiB]'].max()

    print(f"\n--- GPU UTILIZATION REPORT ---")
    print(f"File: {csv_file}")
    print(f"AVG GPU Compute Utilization: {avg_gpu:.2f}%")
    print(f"AVG Memory Utilization: {avg_mem:.2f}%")
    print(f"Peak VRAM Usage: {max_mem_used} MiB")
    
    if avg_gpu < 50 and avg_mem > 80:
        print("DIAGNOSIS: MEMORY BOUND (Adding more batch size won't help)")
    elif avg_gpu > 80:
        print("DIAGNOSIS: COMPUTE BOUND (You are fully utilizing your Tensor Cores)")
    else:
        print("DIAGNOSIS: NETWORK/HOST BOUND (The GPU is waiting on data)")

if __name__ == "__main__":
    analyze_logs(sys.argv)

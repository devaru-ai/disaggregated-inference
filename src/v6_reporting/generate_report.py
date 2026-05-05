import json
import logging
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("TCO-Report")

def generate_tco_report(log_path: str):
    logger.info("📊 Aggregating Telemetry Data...")
    
    with open(log_path, 'r') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    # Calculate Aggregates
    summary = df.groupby('engine').agg(
        avg_ttft_ms=('ttft_ms', 'mean'),
        avg_tpot_ms=('tpot_ms', 'mean'),
        max_vram_gb=('vram_usage_gb', 'max'),
        total_tokens=('total_tokens', 'sum')
    ).round(2)
    
    # Terminal Output
    logger.info("\n========================================================")
    logger.info("   FINAL ARCHITECTURE BENCHMARK REPORT (A100 vs T4)     ")
    logger.info("========================================================")
    logger.info(summary.to_string())
    logger.info("========================================================\n")
    
    # Generate Chart
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot TPOT (Lower is better)
    color = 'tab:red'
    ax1.set_xlabel('Inference Engine')
    ax1.set_ylabel('Avg Time Per Output Token (ms)', color=color)
    ax1.bar(summary.index, summary['avg_tpot_ms'], color=color, alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Plot VRAM Usage on secondary axis
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Peak VRAM Usage (GB)', color=color)  
    ax2.plot(summary.index, summary['max_vram_gb'], color=color, marker='o', linewidth=3)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Engine Performance vs. Resource Utilization')
    fig.tight_layout()
    
    # Save chart to disk instead of trying to pop up a window on a headless server
    output_img = "logs/tco_benchmark_chart.png"
    plt.savefig(output_img)
    logger.info(f"✅ Visual chart saved to {output_img}")

if __name__ == "__main__":
    # Ensure you have pip installed pandas and matplotlib!
    generate_tco_report("logs/request_trace.json")
#!/bin/bash

# Configuration defaults
TARGET=${1:-"monolithic"}
CONCURRENCY=${2:-10}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="logs/report_${TARGET}_c${CONCURRENCY}_${TIMESTAMP}.txt"

export CONCURRENCY=$CONCURRENCY

if [ "$TARGET" == "disaggregated" ]; then
    # You will paste your Colab Ngrok URL here when prompted
    read -p "Enter your Colab Ngrok URL (e.g., https://xyz.ngrok-free.dev): " NGROK_URL
    export WORKER_URL=$NGROK_URL
    echo "Running Disaggregated Test against $WORKER_URL" | tee -a "$REPORT_FILE"
else
    export WORKER_URL="http://localhost:8000"
    echo "Running Monolithic Test against Local A100" | tee -a "$REPORT_FILE"
fi

echo "Concurrency Level: $CONCURRENCY" | tee -a "$REPORT_FILE"
echo "Saving report to: $REPORT_FILE"
echo "==================================================" | tee -a "$REPORT_FILE"

# Run the python script, pipe output to both terminal and file
PYTHONPATH=. python src/v3_disaggregated/7_benchmark_master.py 2>&1 | tee -a "$REPORT_FILE"

echo "==================================================" | tee -a "$REPORT_FILE"
echo "✅ Run complete. Data safely stored in $REPORT_FILE"
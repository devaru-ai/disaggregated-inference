#!/bin/bash
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/gpu_profile_${TIMESTAMP}.csv"

echo "Logging A100 Hardware metrics to ${LOG_FILE}..."
echo "Press Ctrl+C to stop logging."

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 1 > "${LOG_FILE}"
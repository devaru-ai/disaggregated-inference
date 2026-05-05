#!/bin/bash
echo "Starting vLLM Monolithic Engine on A100..."
python -m vllm.entrypoints.openai.api_server \
    --model casperhansen/llama-3-8b-instruct-awq \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --quantization awq
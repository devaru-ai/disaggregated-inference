#!/bin/bash
echo "Starting SGLang Monolithic Engine on A100..."
python -m sglang.launch_server \
    --model-path casperhansen/llama-3-8b-instruct-awq \
    --port 8000 \
    --mem-fraction-static 0.9
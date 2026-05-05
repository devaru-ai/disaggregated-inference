#!/bin/bash
# WARNING: This script takes ~30 minutes to run and requires the TensorRT-LLM docker container.

MODEL_DIR="/raid/home/devai/models/Meta-Llama-3-8B-Instruct"
ENGINE_DIR="/raid/home/devai/engines/Llama-3-8B-TRT-A100"

echo "========================================="
echo "🛠️ INITIATING TENSORRT-LLM BUILD PROCESS"
echo "Target: NVIDIA A100 (SM80)"
echo "========================================="

# Step 1: Convert HuggingFace weights to TRT-LLM Checkpoint format
echo "[1/2] Converting Weights..."
python /app/tensorrt_llm/examples/llama/convert_checkpoint.py \
    --model_dir ${MODEL_DIR} \
    --output_dir /tmp/tllm_checkpoint \
    --dtype bfloat16

# Step 2: Build the highly optimized Execution Engine
echo "[2/2] Compiling C++ Execution Engine..."
trtllm-build \
    --checkpoint_dir /tmp/tllm_checkpoint \
    --output_dir ${ENGINE_DIR} \
    --gemm_plugin bfloat16 \
    --max_batch_size 128 \
    --max_input_len 4096 \
    --max_output_len 2048 \
    --use_paged_context_fmha enable

echo "✅ Build Complete! Engine saved to ${ENGINE_DIR}"
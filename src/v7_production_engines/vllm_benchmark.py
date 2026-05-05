import time
import numpy as np
from vllm import LLM, SamplingParams
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("vLLM-Bench")

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

def run_vllm_benchmark():
    logger.info(f"🚀 Initializing vLLM Engine with {MODEL_ID}...")
    # This will consume heavily from your A100 VRAM
    llm = LLM(model=MODEL_ID, trust_remote_code=True, gpu_memory_utilization=0.85)
    
    prompts = [
        "Explain the architecture of a Disaggregated KV Cache in 500 words.",
        "Write a Python script to perform matrix multiplication.",
        "What are the performance differences between PagedAttention and RadixAttention?",
        "Summarize the history of GPUs in machine learning."
    ]
    
    # We force it to generate exactly 256 tokens so we can measure throughput fairly
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True)
    
    logger.info("🔥 Firing workload batch...")
    start_time = time.perf_counter()
    
    # vLLM batches these automatically
    outputs = llm.generate(prompts, sampling_params)
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    total_tokens = sum([len(output.outputs.token_ids) for output in outputs])
    throughput = total_tokens / total_time
    
    logger.info("=========================================")
    logger.info("📈 vLLM BENCHMARK RESULTS")
    logger.info("=========================================")
    logger.info(f"Total Time:       {total_time:.2f} seconds")
    logger.info(f"Total Tokens:     {total_tokens}")
    logger.info(f"Throughput:       {throughput:.2f} tokens/second")
    logger.info("=========================================")

if __name__ == "__main__":
    run_vllm_benchmark()
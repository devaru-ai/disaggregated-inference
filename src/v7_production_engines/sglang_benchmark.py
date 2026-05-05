import time
import logging
import sglang as sgl

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("SGLang-Bench")

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

@sgl.function
def text_qa(s, question):
    s += sgl.system("You are a senior AI infrastructure engineer.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=256))

def run_sglang_benchmark():
    logger.info(f"🚀 Initializing SGLang Runtime with {MODEL_ID}...")
    
    # Setup SGLang local backend
    backend = sgl.RuntimeEndpoint("http://localhost:30000")
    sgl.set_default_backend(backend)
    
    questions = [
        "Explain the architecture of a Disaggregated KV Cache in 500 words.",
        "Write a Python script to perform matrix multiplication.",
        "What are the performance differences between PagedAttention and RadixAttention?",
        "Summarize the history of GPUs in machine learning."
    ]
    
    logger.info("🔥 Firing workload batch (Watch the Prefix Cache hit rate!)...")
    start_time = time.perf_counter()
    
    # Execute heavily concurrent batch
    states = text_qa.run_batch([{"question": q} for q in questions], num_threads=4)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Estimate tokens based on max_tokens setting for benchmark baseline
    total_tokens = 256 * len(questions) 
    throughput = total_tokens / total_time
    
    logger.info("=========================================")
    logger.info("📈 SGLANG BENCHMARK RESULTS")
    logger.info("=========================================")
    logger.info(f"Total Time:       {total_time:.2f} seconds")
    logger.info(f"Est. Throughput:  {throughput:.2f} tokens/second")
    logger.info("=========================================")

if __name__ == "__main__":
    logger.warning("NOTE: You must start the SGLang server in a separate terminal first!")
    logger.warning("Command: python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000")
    # run_sglang_benchmark() # Uncomment when server is running
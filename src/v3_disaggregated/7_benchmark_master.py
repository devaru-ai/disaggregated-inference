import asyncio
import time
import httpx
import json
import statistics
import os

# ==========================================
# BENCHMARK CONFIGURATION
# ==========================================
# Reads from the terminal environment, defaults to localhost if not set
WORKER_URL = os.getenv("WORKER_URL", "http://localhost:8000")
MODEL_NAME = "casperhansen/llama-3-8b-instruct-awq"
CONCURRENCY_LEVEL = int(os.getenv("CONCURRENCY", "10"))

# ==========================================
# 1. PURE LLM WORKLOAD (Decode Bound)
# ==========================================
PURE_LLM_PROMPT = "Explain the architectural differences between Hopper and Ampere GPUs in extreme detail."

# ==========================================
# 2. ROUTER AGENT WORKLOAD (Fast Classification)
# ==========================================
AGENT_ROUTER_PROMPT = """You are a Routing Agent. Classify the user query into one of these queues: [HARDWARE, SOFTWARE, NETWORK].
Respond ONLY in JSON: {"queue": "YOUR_CHOICE", "confidence": 0.0-1.0}

USER QUERY: "Why is my NCCL ring throwing a timeout error across nodes?"
"""

# ==========================================
# 3. RETRIEVAL AGENT WORKLOAD (Prefill Bound - Massive Prompt)
# ==========================================
# We simulate a heavy context window by repeating a block of text to test prompt caching (SGLang's superpower)
DUMMY_CONTEXT = "The system achieved a TTFT of 45ms and a TPOT of 12ms under a load of 64 concurrent users. " * 50
AGENT_RETRIEVAL_PROMPT = f"""You are a Data Extraction Agent. Read the following context and extract the latency metrics.
Respond ONLY in JSON: {{"ttft_ms": int, "tpot_ms": int, "max_users": int}}

CONTEXT:
{DUMMY_CONTEXT}

Extract the metrics now:
"""

async def fire_single_request(client: httpx.AsyncClient, payload: dict, req_id: int):
    """Fires a single streaming request and tracks its specific metrics."""
    start_time = time.time()
    first_token_time = None
    token_count = 0

    # Required to bypass Ngrok's interstitial warning page when testing Disaggregated
    headers = {"ngrok-skip-browser-warning": "true"}

    try:
        async with client.stream("POST", f"{WORKER_URL}/v1/completions", json=payload, headers=headers) as response:
            if response.status_code != 200:
                return {"error": response.status_code}

            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    if first_token_time is None:
                        first_token_time = time.time()
                    
                    try:
                        # Parse to ensure it's valid JSON; we don't need to save the actual text for the load test
                        json.loads(line[6:])
                        token_count += 1
                    except (KeyError, json.JSONDecodeError):
                        continue
    except Exception as e:
        return {"error": str(e)}

    end_time = time.time()
    
    ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
    generation_time = end_time - first_token_time if first_token_time else 0
    tpot = (generation_time / token_count) * 1000 if token_count > 0 else 0

    return {
        "ttft": ttft,
        "tpot": tpot,
        "tokens": token_count,
        "total_time": end_time - start_time
    }

async def run_concurrent_benchmark(workload_name: str, prompt: str, max_tokens: int):
    print(f"\n" + "="*60)
    print(f"🚀 STARTING: {workload_name}")
    print(f"⚙️  Concurrency Level: {CONCURRENCY_LEVEL} simultaneous requests")
    print(f"🌐 Target Node: {WORKER_URL}")
    print("="*60)
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": True
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Create N tasks to fire simultaneously
        tasks = [fire_single_request(client, payload, i) for i in range(CONCURRENCY_LEVEL)]
        
        # Hit the global stopwatch
        global_start = time.time()
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        global_end = time.time()

    # Process Results
    successful_runs = [r for r in results if "error" not in r and r.get("tokens", 0) > 0]
    errors = len(results) - len(successful_runs)
    
    if not successful_runs:
        print(f"❌ ALL REQUESTS FAILED. Server might be OOM or unreachable.")
        return

    # Sort the list immediately to guarantee numerical order
    ttfts = sorted([float(r["ttft"]) for r in successful_runs])
    tpots = [float(r["tpot"]) for r in successful_runs]
    total_tokens = sum(r["tokens"] for r in successful_runs)
    global_duration = float(global_end - global_start)
    throughput = total_tokens / global_duration

    # --- BULLETPROOF P95 MATH ---
    # Find the index that represents 95% of the list length
    p95_index = int(len(ttfts) * 0.95)
    # Ensure it never goes out of bounds, even with small concurrency
    p95_index = min(p95_index, len(ttfts) - 1)
    p95_ttft = ttfts[p95_index] if ttfts else 0.0

    print(f"✅ Benchmark Complete ({len(successful_runs)}/{CONCURRENCY_LEVEL} succeeded, {errors} errors)")
    print(f"Total Wall-Clock Time : {global_duration:.2f} sec")
    print(f"Total System Throughput: {throughput:.2f} tokens/sec")
    print("-" * 60)
    print(f"AVG TTFT (Time to First Token) : {statistics.mean(ttfts):.2f} ms")
    print(f"P95 TTFT (95th Percentile)     : {p95_ttft:.2f} ms")
    print(f"AVG TPOT (Time Per Token)      : {statistics.mean(tpots):.2f} ms")
    print("="*60 + "\n")

async def main():
    # 1. Pure LLM: High max_tokens, letting it generate a lot of text.
    await run_concurrent_benchmark("PURE LLM WORKLOAD", PURE_LLM_PROMPT, max_tokens=300)
    
    # 2. Router Agent: Short generation, needs to be extremely fast.
    await run_concurrent_benchmark("AGENT 1: FAST ROUTER", AGENT_ROUTER_PROMPT, max_tokens=20)

    # 3. Retrieval Agent: Massive prefill prompt, tests prompt caching.
    await run_concurrent_benchmark("AGENT 2: HEAVY RETRIEVAL", AGENT_RETRIEVAL_PROMPT, max_tokens=50)

if __name__ == "__main__":
    asyncio.run(main())
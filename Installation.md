# Installation
Run this once to prep your environment.

```bash
git clone https://github.com/devaru-ai/disaggregated-inference.git
cd disaggregated-inference
docker build -t disagg-inference:latest .
```

# Monolithic Serving
This phase tests a standard, single-GPU deployment. We test both `vLLM` (standard) and `SGLang` (RadixAttention) to compare their baseline throughput.

## Case A: Monolithic with vLLM

### Terminal 1 (Start the Engine):
```bash
docker run --name vllm-engine --gpus '"device=0"' --shm-size=16g -p 8000:8000 --rm -it disagg-inference:latest \
  python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8000 \
  --enforce-eager
```

### **Terminal 2:**
*Wait for Terminal 1 to say "Uvicorn running on [http://0.0.0.0:8000](http://0.0.0.0:8000)", then run:*
```bash
docker exec -it vllm-engine /bin/bash -c "cd /content/disaggregated-inference && export CONCURRENCY=256 && ./scripts/run_tests.sh monolithic"
```

## **Terminal 3:**
*Run this on your host machine exactly while Terminal 2 is running:*
```bash
nvidia-smi dmon -s uc -c 20 > logs/monolithic_vllm_dmon.txt
```
*(When finished, press `Ctrl+C` in Terminal 1 to kill the server).*



## Case B: Monolithic with SGLang
### **Terminal 1 (Start the Engine):**
```bash
docker run --name sglang-engine --gpus '"device=0"' --shm-size=16g -p 8000:8000 --rm -it disagg-inference:latest \
  python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8000
```

### **Terminal 2:**
*Wait for Terminal 1 to load, then run:*
```bash
docker exec -it sglang-engine /bin/bash -c "cd /content/disaggregated-inference && export CONCURRENCY=256 && ./scripts/run_tests.sh monolithic"
```

### **Terminal 3:**
```bash
nvidia-smi dmon -s uc -c 20 > logs/monolithic_sglang_dmon.txt
```
*(When finished, press `Ctrl+C` in Terminal 1 to kill the server).*



# Disaggregated Serving
This phase splits the workload. The **Router (Univ A100)** acts as the lightweight traffic cop, and the **Worker (Colab T4 Edge)** does the heavy lifting over the network.

### Step 1: Start the Remote Worker (Edge Node)
**Terminal 1 (On the Edge Machine / Colab T4):**
Ensure your Ngrok tunnel is running, then start the worker container:
```bash
docker run --name worker-node --gpus '"device=0"' --rm -it disagg-inference:latest \
  ./scripts/run_tests.sh disaggregated-worker
```

### Step 2: Start the Router & Orchestrate
**Terminal 2 (On A100):**
This container acts as the Master. It takes the requests and routes them to the Worker URL.
```bash
docker run --name router-node --gpus '"device=0"' --rm -it disagg-inference:latest \
  ./scripts/run_tests.sh disaggregated-router
```
*(You will be prompted to paste your Worker's Ngrok URL here).*

### Step 3: Prove the Router is Idle (The Telemetry Trap)
**Terminal 3 (On A100):**
While Terminal 2 is routing traffic, run this to prove the A100's compute is sitting at exactly 0.0%:
```bash
nvidia-smi dmon -s uc -c 20 > logs/disaggregated_router_idle_proof.txt
```

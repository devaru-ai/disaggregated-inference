# Disaggregated Inference: Hardware Benchmarks 

This document outlines the bare-metal profiling and performance benchmarks for a distributed LLM inference pipeline. The tests compare a traditional centralized architecture (Monolithic A100) against an edge-cloud split architecture (Disaggregated A100 Orchestrator -> T4 Edge Worker via Ngrok).

**Testing Environment:**
* **Model:** Llama-3-8B-AWQ (4-bit precision)
* **Concurrency:** 10 simultaneous requests
* **Engines Tested:** vLLM, SGLang

## Table 1: The Architecture Baseline (Monolithic vs. Disaggregated)

**Objective:** Isolate the "Network Tax" of bouncing requests over a public internet tunnel (Ngrok) to a remote, lower-tier GPU compared to a 0-network-latency monolithic floor.

| Architecture | Hardware Setup | Agent Workload | TTFT (ms) | TPOT (ms/tok) | Throughput (tok/s) | Latency Delta (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Monolithic** | A100 (Single Node) | Router (Light Prompt) | 123.68 | 14.64 | 477.91 | 0 (Internal) |
| **Monolithic** | A100 (Single Node) | Retrieval (Heavy Prompt) | 934.08 | 21.19 | 248.58 | 0 (Internal) |
| **Disaggregated** | A100 Master -> T4 Worker | Router (Light Prompt) | 1008.23 | 246.08 | 32.60 | +884.55 |
| **Disaggregated** | A100 Master -> T4 Worker | Retrieval (Heavy Prompt) | 1828.04 | 261.23 | 32.99 | +893.96 |

**Architectural Interpretation:** Routing requests over an Ngrok tunnel to a remote Turing-class T4 adds roughly **~884ms of latency** to the Time-To-First-Token (TTFT), and drastically increases the Time-Per-Output-Token (TPOT). This mathematically proves that while edge nodes are highly viable for asynchronous background workloads, this specific network configuration is fatal for real-time, user-facing latency.


## Table 2: The Inference Engine Comparison (Monolithic A100)

**Objective:** Freeze the hardware (A100) and model, and benchmark engine-level optimizations specifically against Agentic workflows (which rely on repetitive, heavy system prompts).

| Inference Engine | Core Optimization | Agent Workload | TTFT (ms) | TPOT (ms/tok) | Concurrent Load | VRAM Used |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **vLLM** *(Baseline)* | PagedAttention | Retrieval Agent | 934.08 | 21.19 | 10 | Dynamic |
| **SGLang** | RadixAttention (Caching) | Retrieval Agent | 417.91 | 17.32 | 10 | 75.2 GB (Static) |

**Architectural Interpretation:** This proves the immense value of **RadixAttention** on bare metal. By caching the massive system prompt across the 10 concurrent requests, SGLang slashed the Heavy Retrieval TTFT from 934.08 ms down to 417.91 ms—a **55% reduction in prefill latency**. This demonstrates how software/framework optimization can protect hardware from choking under heavy context windows.


## Table 3: System Profiling & Hardware Saturation

**Objective:** Analyze the physical state of the metal during the benchmark to identify the true system bottlenecks (Compute vs. Memory vs. Network).

| Architecture Role | Hardware | Test Batch Size | VRAM Allocation | SM Compute Peak | Primary Bottleneck |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Monolithic Engine** | Univ A100 | 10 | 94% (75.2/80GB) | 0.20% | **I/O Bound** (GPU is too fast; needs 100+ concurrency to wake up) |
| **Disaggregated Worker** | Colab T4 | 10 | 95% (14.2/15GB) | 40.0% | **Host/Network Bound** (Ngrok network jitter prevents optimal batching) |
| **Disaggregated Router** | Univ A100 | 10 | 0% (0/80GB) | 0.0% | **Network Bound** (CPU just waiting on Edge node responses) |

**Architectural Interpretation:** Theoretical assumptions suggested the older T4 would cap out at 99% SM compute under load. However, the telemetry revealed it peaked at only **40% compute**. 

This exposes the **Drip-Feed Effect**: Because requests traveled over the public internet via Ngrok, latency jitter caused them to arrive staggered. The vLLM engine couldn't group them into a single, massive, highly efficient batch. Instead, the T4 processed micro-batches, spinning its cores to 40%, finishing the math, and waiting for the next packet. The edge GPU wasn't compute-bound; it was host/network-bound, starved by the tunnel.

# Next Up

## 1. Establishing the Peak Performance on Monolithic H100
While the A100 benchmarking proved that the pipeline is currently I/O-bound, the A100 is ultimately constrained by its memory bandwidth. The next step is to run the exact same Monolithic benchmark suite on an NVIDIA H100 (SM90 architecture). 
### Objective:
- Establish the absolute ceiling for throughput (Tokens/Sec) and observe the behavior of FP8/BF16 kernels when memory bandwidth is no longer the primary bottleneck. 
- This will serve as the "Gold Standard" baseline against which all future distributed or heterogeneous cluster topologies (e.g., H100 Prefill + A100 Decode) will be measured.

## 2. Benchmark TensorRT-LLM
To complete the Inference Engine analysis, the pipeline will be stress-tested against Nvidia’s native TensorRT-LLM. While vLLM provided the baseline and SGLang proved the value of RadixAttention caching, TRT-LLM represents the "Final Boss" of raw inference speed.
### Requirements:
- This requires Ahead-Of-Time (AOT) C++ compilation to fuse the model weights and operators into a highly optimized `.engine` file specific to the target silicon.
- Complete the engine benchmarking matrix by proving how bare-metal C++ graph compilation impacts Time-Per-Output-Token (TPOT) compared to standard Python-based inference servers. 

## **3. Deploy Mixture of Experts (MoE)**
Currently, the orchestrator routes entire prompts to homogeneous dense models. The next step for this disaggregated pipeline is to introduce Sparse Architectures (Mixture of Experts, such as Mixtral).
### Objective:
- Evaluate how MoE models perform across a disaggregated network where token-level expert routing adds another layer of scheduling complexity.
- Determine if the memory-saving benefits of sparse active parameters outweigh the orchestration overhead, and explore the potential of physically distributing different "Experts" across different edge nodes in the cluster.

## 4. 70B+ Models & Multi-GPU Parallelism
To truly stress-test the orchestrator's routing and backpressure mechanisms, the pipeline must be scaled beyond models that fit onto a single accelerator. The next architectural leap involves deploying a 70B+ parameter model (e.g., Llama-3-70B) across the cluster.
- Implement Tensor Parallelism (TP) to shard the model weights across multiple GPUs, or explore Pipeline Parallelism (PP) to split the model's layers across the edge network.
- **The Topologies:** To maintain an apples-to-apples comparison, this massive scale-out will be benchmarked across three specific configurations:
  * **Monolithic A100 (Multi-GPU):** Establishing the standard NVLink/NCCL communication baseline on Ampere architecture.
  * **Monolithic H100 (Multi-GPU):** The absolute high-bandwidth "Gold Standard" for tensor parallelism.
  * **Heterogeneous Disaggregated (H100 Master -> A100 Worker):** This measures how high-speed internal multi-GPU communication (TP on the H100s) interacts with the external network latency of pushing the resulting KV caches to an A100 edge node.
- Measure how intra-node communication overhead (NVLink/NCCL) interacts with inter-node internet latency (Ngrok), and mathematically prove that the 62GB backpressure safety valves hold up when dealing with massive, distributed KV cache allocations.

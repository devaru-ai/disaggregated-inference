# Dynamo-Inspired Disaggregated Inference Platform

#### A heterogeneous disaggregated inference prototype inspired by NVIDIA Dynamo for benchmarking prefill and decode performance across GPU tiers and evaluating LLM serving trade-offs.

### What is NVIDIA Dynamo?

NVIDIA Dynamo is an open-source inference framework designed for serving large-scale generative and reasoning models in distributed GPU environments. It targets high-throughput, low-latency inference by treating model serving as a cluster-level scheduling and memory management problem.

Dynamo’s central architectural idea is **prefill–decode disaggregation**, where prompt processing (prefill) and token generation (decode) are separated and executed on different GPU resources. This allows each stage to be independently optimized for compute, memory, and latency characteristics.

Dynamo is designed to work with modern inference engines such as **vLLM, SGLang, and TensorRT-LLM**, providing a unified orchestration layer across them.

### Key Architectural Concepts

* **KV-aware routing:** Requests are routed based on KV-cache locality to reduce redundant computation and improve cache reuse.
* **Cluster-level scheduling:** GPU resources are dynamically allocated based on workload demand and service constraints.
* **KV cache management across memory tiers:** KV states can be moved or offloaded across memory hierarchies to reduce GPU memory pressure.
* **High-performance interconnect utilization:** Efficient communication between nodes is required to support disaggregated execution at scale.


# Our Architecture

We evaluate LLM inference performance across monolithic and disaggregated serving topologies, isolating compute, memory, and network bottlenecks. The system is designed to mirror key architectural ideas from NVIDIA Dynamo (prefill–decode disaggregation, KV-aware routing, and cluster-level scheduling), but implemented as a heterogeneous experimental prototype across GPU tiers.


## 1. Monolithic Serving Baselines 

We first establish performance baselines using single-node inference, where both prefill and decode execute on the same GPU, defining compute- and memory-bound ceilings.

* **Hardware:** A100 and H100 single-node configurations (Ampere vs Hopper comparison, including FP8 / Transformer Engine behavior on H100)
* **Engines:** vLLM (PagedAttention) and SGLang (RadixAttention, KV-aware routing, prefix matching, KV cache reuse)
* **Extended Engine:** TensorRT-LLM for compiled execution graphs, kernel fusion, and FP8-optimized Transformer Engine kernels on H100

**Objective:** Define compute- and memory-bound ceilings before introducing distributed execution.

SGLang is used to isolate KV-cache reuse, prefix matching, and prompt caching effects on TTFT under repeated system prompt workloads.



## 2. Role-Based Disaggregation

We break the monolithic execution path into an orchestrator–worker split to quantify distributed inference overhead.

### 2.1 Topology

* **Router:** A100 (orchestrator only, 0% SM utilization, API routing + load balancing)
* **Worker:** T4 edge GPU (full forward execution: prefill + decode)
* **Transport:** Public internet tunnel (Ngrok, low-bandwidth, high-latency simulation)

### Objective

Measure the **network tax** introduced when KV state, prompts, and intermediate tensors traverse a constrained interconnect.

### Execution Model

The A100 only handles request routing and scheduling. The T4 executes the full forward pass (prefill + decode), simulating a naïve disaggregated serving system.

This stresses cross-node communication overhead and approximates production systems that require **NVIDIA NIXL**, NVLink, InfiniBand, and KV-aware routing with locality awareness.

**Key insight:** The system becomes network-bound rather than compute-bound, and batching efficiency collapses due to jitter-induced micro-batching.


## 3. Phase-Based Disaggregation (Dynamo-Aligned Topology)

We extend the system to a phase-split execution model inspired by NVIDIA Dynamo, explicitly separating prefill and decode phases and introducing KV-cache transfer as a first-class system primitive.

### 3.1 Topology

* **Prefill:** H100 (compute-bound, Transformer Engine, FP8 acceleration, KV-cache generation)
* **Decode:** A100 (memory-bound token generation)
* **Interconnect:** Physical high-speed KV-cache transfer path between nodes 

### 3.2 Execution Model

* H100 performs compute-heavy prefill and generates KV cache
* KV cache is transferred across nodes (state handoff layer)
* A100 performs memory-bound decode

**Objective:** Map compute-heavy prefill to H100 and memory-bound decode to A100, isolating phase-specific hardware efficiency and KV transfer cost.

This mirrors Dynamo’s core design principle: disaggregating inference phases to independently optimize compute, memory, and communication.


## 4. Model Scaling & Parallelism Strategies

We evaluate system behavior across increasing model sizes and execution strategies to observe bottleneck transitions.

### 4.1 Model Spectrum

* 8B (Llama-class baseline)
* 20B (intermediate stress)
* 70B (large-scale memory + communication stress)

**Objective:** Observe transitions across compute-bound → memory-bound → communication-bound regimes.


### 4.2 Mixture of Experts (MoE)

We evaluate sparse activation models to measure:

* Expert routing overhead
* VRAM footprint of multiple resident experts
* Latency cost of conditional computation
* KV-cache interaction with expert selection


### 4.3 Parallelism

* **Tensor Parallelism (TP):** Intra-node weight sharding across GPUs
* **Expert Parallelism (EP):** Distributed placement of MoE experts across nodes

**Objective:** Evaluate scaling efficiency under both intra-node and inter-node decomposition.


## 5. Evaluation Workloads

We stress the system across three workload classes:

* **LLM inference workloads:** throughput (tokens/sec), TTFT, TPOT under concurrency
* **Retrieval-heavy agents:** KV-cache reuse efficiency and prefix matching effectiveness
* **Multi-turn reasoning:** context switching overhead and router stability under high-frequency requests




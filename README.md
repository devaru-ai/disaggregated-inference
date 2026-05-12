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


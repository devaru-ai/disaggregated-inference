# Disaggregated Inference Execution Trace

## Day 0–2: Distributed Inference System Buildout and Validation

| Execution Phase | Command  | Node / Terminal | Outcome |
| :--- | :--- | :--- | :--- |
| **1. Hardware Profiling** | `CUDA_VISIBLE_DEVICES=1 python src/v2_monolithic/5_mono_profiler.py` | A100 (Local) | **OOM Ceiling Discovery:** Pushed the A100 until it crashed at 78GB to find the absolute physical limit of the VRAM. This number is used to mathematically derive the 62GB safety backpressure threshold. |
| **2. Environment Prep** | `conda activate infra-platform` | All Terminals | **Dependency Isolation:** Ensures the scripts are running against the "Golden Standard" PyTorch 2.4.0 environment rather than the system default. |
| **3. Control Plane: Master** | `python src/infrastructure/scheduler/heartbeat_listener.py` | Terminal 1 | **Listener Initialization:** Boots up the Master node's receiver to listen for health checks and incoming tensor payloads. |
| **4. Control Plane: Worker** | `python src/infrastructure/transport/heartbeat.py` | Terminal 2 | **Liveness Probe:** Simulates a remote worker sending an asynchronous "I'm alive" pulse every second to establish the control plane. |
| **5. Data Plane Test** | `python -c "from src.infrastructure... client.send_tensor(...)"` | Terminal 3 | **RPC Payload Test:** Bypasses GPU allocation by creating a dummy tensor on the CPU and firing it over local network RPC to verify the transport layer works before adding internet latency. |
| **6. Disaggregated Boot** | `PYTHONPATH=. python src/v3_disaggregated/6_node_a_master.py` | Terminal 1 (Master) | **Orchestrator Initialization:** Boots the actual traffic-routing Master node. `PYTHONPATH=.` forces Python to recognize the root directory to fix import pathing errors. |
| **7. Cross-Cloud Firing** | `curl -X POST "http://127.0.0.1:8080/generate?prompt=TestingColabWorker"` | Terminal 3 (Client) | **End-to-End Verification:** Fires an HTTP payload locally. The Master catches it, routes it through the Ngrok tunnel over the public internet, and hits the Colab T4 worker for remote processing. |
| **8. Load Testing** | `./scripts/run_tests.sh disaggregated 10` | Terminal 3 (Client) | **Concurrency Benchmarking:** Fires a burst of 10 concurrent requests at the pipeline to measure real-world performance metrics (TTFT, TPOT) and expose hardware/network bottlenecks under load. |


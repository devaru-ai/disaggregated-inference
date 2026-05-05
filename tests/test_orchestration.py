import asyncio
from src.v4_orchestration.kv_router import PrefixRouter
from src.v4_orchestration.hetero_tasks import TaskDispatcher, HardwareRequirement
from src.v4_orchestration.agents.research_agent import ResearchAgent
from src.v4_orchestration.agents.coding_agent import CodingAgent


async def main():
    print("\n--- INITIALIZING ORCHESTRATOR ---")
    # Simulate your Colab Worker and a local backup worker
    router = PrefixRouter(["Worker_T4_Colab", "Worker_A100_Local"])
    dispatcher = TaskDispatcher()
    
    researcher = ResearchAgent()
    coder = CodingAgent()

    print("\n--- SIMULATING INCOMING WORKLOADS ---")
    
    # Workload 1: Web Search (CPU)
    await dispatcher.submit_task("W1", HardwareRequirement.CPU_ONLY, {"tool": "scrape_wiki"})
    
    # Workload 2: Research Inference (GPU)
    res_job = researcher.execute_tool_chain("Disaggregated memory architectures")
    target_node = router.get_worker(res_job["routing_key"])
    await dispatcher.submit_task("W2", HardwareRequirement.GPU_REQUIRED, {"node": target_node, "data": res_job})

    # Workload 3: Coding Inference (GPU)
    code_job = coder.generate_review_workload("def foo(): pass")
    target_node = router.get_worker(code_job["routing_key"])
    await dispatcher.submit_task("W3", HardwareRequirement.GPU_REQUIRED, {"node": target_node, "data": code_job})

    # Workload 4: ANOTHER Research Inference (GPU) - Should hit the exact same node as Workload 2!
    res_job_2 = researcher.execute_tool_chain("Prefix caching in vLLM")
    target_node_2 = router.get_worker(res_job_2["routing_key"])
    await dispatcher.submit_task("W4", HardwareRequirement.GPU_REQUIRED, {"node": target_node_2, "data": res_job_2})

if __name__ == "__main__":
    asyncio.run(main())
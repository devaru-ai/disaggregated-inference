import logging
import asyncio
from enum import Enum

logger = logging.getLogger("Hetero-Dispatcher")

class HardwareRequirement(Enum):
    CPU_ONLY = "CPU"
    GPU_REQUIRED = "GPU"

class TaskDispatcher:
    def __init__(self):
        self.cpu_queue = asyncio.Queue()
        self.gpu_queue = asyncio.Queue()

    async def submit_task(self, task_id: str, requirement: HardwareRequirement, payload: dict):
        if requirement == HardwareRequirement.CPU_ONLY:
            logger.info(f"[DISPATCH] Task {task_id} -> CPU Queue (I/O, Parsing, Web Search)")
            await self.cpu_queue.put(payload)
        else:
            logger.info(f"[DISPATCH] Task {task_id} -> GPU Queue (LLM Inference, Tensor Math)")
            await self.gpu_queue.put(payload)

    async def get_next_gpu_task(self):
        return await self.gpu_queue.get()
        
    async def get_next_cpu_task(self):
        return await self.cpu_queue.get()
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Scheduler-Backpressure")

class BackpressureController:
    def __init__(self, vram_limit_gb=62.0):
        # We use the 62GB limit calculated from our 78GB A100 ceiling
        self.vram_limit_gb = vram_limit_gb
        self.vram_limit_bytes = vram_limit_gb * (1024 ** 3)
        self.active_requests = 0

    def can_accept_request(self) -> bool:
        if not torch.cuda.is_available():
            return True # Fallback for CPU testing
            
        current_vram = torch.cuda.memory_allocated()
        
        if current_vram >= self.vram_limit_bytes:
            logger.warning(f"CIRCUIT BREAKER: VRAM at {current_vram / (1024**3):.2f}GB. Rejecting new traffic.")
            return False
            
        return True
        
    def add_request(self):
        self.active_requests += 1
        
    def complete_request(self):
        self.active_requests = max(0, self.active_requests - 1)
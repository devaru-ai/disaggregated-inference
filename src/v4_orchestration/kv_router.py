import hashlib
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("KV-Router")

class PrefixRouter:
    def __init__(self, active_workers: List[str]):
        self.workers = active_workers
        if not self.workers:
            raise ValueError("Router requires at least one active worker.")

    def get_worker(self, system_prompt: str) -> str:
        """
        Uses consistent hashing on the prompt prefix.
        Ensures the same system prompt always hits the same node to reuse KV cache.
        """
        # Create a deterministic hash of the prefix
        hash_val = int(hashlib.md5(system_prompt.encode('utf-8')).hexdigest(), 16)
        
        # Modulo routing
        worker_idx = hash_val % len(self.workers)
        selected_worker = self.workers[worker_idx]
        
        logger.info(f"[ROUTER] Prefix Hash: {hash_val % 10000:04d} -> Routed to: {selected_worker}")
        return selected_worker

    def update_workers(self, new_workers: List[str]):
        self.workers = new_workers
        logger.info(f"[ROUTER] Worker pool updated: {len(self.workers)} nodes active.")
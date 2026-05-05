import logging

logger = logging.getLogger("Coding-Agent")

class CodingAgent:
    def __init__(self, system_prompt: str = "You are a senior principal engineer. Review this code for Big-O efficiency."):
        self.system_prompt = system_prompt
        self.routing_prefix = "SYSTEM: " + system_prompt

    def generate_review_workload(self, code_snippet: str):
        logger.info("[CODING AGENT] Compiling syntax tree and formatting prompt...")
        
        compiled_prompt = f"{self.routing_prefix}\nCODE:\n{code_snippet}\nUSER: Optimize this."
        
        return {
            "type": "inference_request",
            "agent": "coding",
            "routing_key": self.routing_prefix,
            "compiled_prompt": compiled_prompt
        }
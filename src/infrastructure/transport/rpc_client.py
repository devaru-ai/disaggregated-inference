import torch
import hashlib
import logging
import json
import time
import requests

# Configure structured JSON logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Transport-RPC")

class RPCClient:
    def __init__(self, node_id):
        self.node_id = node_id

    def calculate_checksum(self, tensor: torch.Tensor) -> str:
        # Checksum ensures data wasn't corrupted in transit
        return hashlib.md5(tensor.cpu().numpy().tobytes()).hexdigest()

    def send_tensor(self, target_url: str, tensor: torch.Tensor):
        checksum = self.calculate_checksum(tensor)
        
        # We send metadata to validate the connection and routing logic.
        payload = {
            "node_id": self.node_id,
            "timestamp": time.time(),
            "checksum": checksum,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype)
        }
        
        logger.info(json.dumps({
            "event": "TENSOR_SEND_INITIATED",
            "target": target_url,
            "checksum": checksum,
            "size_elements": tensor.numel()
        }))
        
        # This specific header tells Ngrok to skip the HTML warning page 
        # and forward our JSON payload directly to the Colab worker.
        headers = {
            "ngrok-skip-browser-warning": "true",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                f"{target_url}/receive_tensor", 
                json=payload, 
                headers=headers,
                timeout=2.0
            )
            if response.status_code == 200:
                logger.info(json.dumps({"event": "TENSOR_SEND_SUCCESS", "target": target_url}))
                return True
            else:
                logger.error(json.dumps({"event": "TENSOR_SEND_FAILED", "status_code": response.status_code}))
                return False
        except Exception as e:
            logger.error(json.dumps({"event": "TENSOR_SEND_ERROR", "error": str(e)}))
            return False
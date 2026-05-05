import time
import requests
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Heartbeat-Worker")

class HeartbeatMonitor:
    def __init__(self, master_url, node_id, interval=1.0):
        self.master_url = master_url
        self.node_id = node_id
        self.interval = interval

    def send_pulse(self):
        try:
            payload = {"node_id": self.node_id, "status": "ALIVE", "timestamp": time.time()}
            # Fast timeout so the worker doesn't hang if the master dies
            response = requests.post(f"{self.master_url}/heartbeat", json=payload, timeout=0.5)
            if response.status_code == 200:
                logger.info(json.dumps({"event": "HEARTBEAT_ACK", "master": self.master_url}))
                return True
        except requests.exceptions.RequestException as e:
            logger.warning(json.dumps({"event": "HEARTBEAT_FAILED", "error": str(e)}))
            return False

    def start(self):
        logger.info(f"Starting heartbeat monitor for {self.node_id} -> {self.master_url}")
        while True:
            self.send_pulse()
            time.sleep(self.interval)

if __name__ == "__main__":
    # When deployed, this points to the Master's IP. For local testing, we use localhost.
    monitor = HeartbeatMonitor("http://127.0.0.1:8080", "Worker_T4_01")
    monitor.start()
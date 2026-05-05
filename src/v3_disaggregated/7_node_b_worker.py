from fastapi import FastAPI, Request
import uvicorn
import logging
import json
import asyncio
import threading
from src.infrastructure.transport.heartbeat import HeartbeatMonitor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Node-B-Worker")

app = FastAPI()

@app.post("/receive_tensor")
async def receive_tensor(request: Request):
    payload = await request.json()
    logger.info(f"Received KV Cache from {payload.get('node_id')}. Starting Decode Phase...")
    
    # Simulate Decode token generation
    for i in range(1, 4):
        await asyncio.sleep(0.2)
        logger.info(f"Generated token {i}...")
        
    return {"status": "SUCCESS", "tokens_generated": 3}

@app.on_event("startup")
def startup_event():
    # Start heartbeat in a background thread to ping the master
    monitor = HeartbeatMonitor("http://127.0.0.1:8080", "Worker_T4_01", interval=2.0)
    threading.Thread(target=monitor.start, daemon=True).start()

if __name__ == "__main__":
    logger.info("Starting Decode Worker on port 8081...")
    # Note: Running on 8081 so it doesn't clash with the Master on 8080
    uvicorn.run(app, host="0.0.0.0", port=8081)
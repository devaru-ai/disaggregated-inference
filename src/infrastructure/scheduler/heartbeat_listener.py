from fastapi import FastAPI, Request
import uvicorn
import logging
import json
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Master-Listener")

app = FastAPI()
worker_health = {}

@app.post("/heartbeat")
async def receive_heartbeat(request: Request):
    data = await request.json()
    node_id = data.get("node_id")
    worker_health[node_id] = time.time()
    logger.info(json.dumps({"event": "HEARTBEAT_RECEIVED", "node_id": node_id}))
    return {"status": "ACK"}

@app.post("/receive_tensor")
async def receive_tensor(request: Request):
    data = await request.json()
    logger.info(json.dumps({"event": "TENSOR_METADATA_RECEIVED", "payload": data}))
    return {"status": "SUCCESS"}

if __name__ == "__main__":
    logger.info("Starting Master Listener on port 8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
import uvicorn
import logging
import httpx
import asyncio
from fastapi import FastAPI, Request

# ==========================================
# DISAGGREGATED INFERENCE CONFIGURATION
# ==========================================
# Update this URL every time you restart the Colab Ngrok tunnel
COLAB_URL = "https://attendant-capacity-uncharted.ngrok-free.dev" 

# Must match the exact quantized model hosted on the worker node
MODEL_NAME = "casperhansen/llama-3-8b-instruct-awq"

# ==========================================
# LOGGING SETUP
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - Master-Orchestrator - %(levelname)s - %(message)s')
logger = logging.getLogger("MASTER")

app = FastAPI()
gpu_queue = asyncio.Queue()

# ==========================================
# BACKGROUND INFERENCE ENGINE
# ==========================================
async def process_gpu_queue():
    """Background loop that executes the cross-node LLM calls."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        while True:
            task = await gpu_queue.get()
            prompt = task["prompt"]
            
            logger.info(f"[PIPELINE] Routing research task to Engine: {prompt[:50]}...")
            
            payload = {
                "model": MODEL_NAME,
                "prompt": prompt,
                "max_tokens": 300,  # Safely fits inside your Colab's 512 context window
                "temperature": 0.1
            }
            
            # Required to bypass Ngrok's interstitial warning page
            headers = {"ngrok-skip-browser-warning": "true"}
            
            try:
                response = await client.post(
                    f"{COLAB_URL}/v1/completions", 
                    json=payload, 
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Bulletproof parsing block
                    try:
                        if "choices" not in data or not isinstance(data["choices"], list):
                            raise ValueError(f"Invalid response schema: {data}")

                        if len(data["choices"]) == 0:
                            raise ValueError("Empty choices array")

                        # ✅ THE CRITICAL FIX: Grab the first dictionary out of the list
                        choice = data["choices"][0]   

                        if isinstance(choice, dict) and "text" in choice:
                            text = choice["text"]  # /v1/completions
                        elif isinstance(choice, dict) and "message" in choice:
                            text = choice["message"]["content"]  # /v1/chat/completions
                        else:
                            raise ValueError(f"Unknown choice format: {choice}")
                        
                        # Handle the truncation warning gracefully
                        finish_reason = choice.get("finish_reason")
                        if finish_reason == "length":
                            logger.warning("⚠️ Output truncated (max_tokens reached)")

                        logger.info("\n" + "="*50)
                        logger.info("✅ AGENT RESPONSE (research):")
                        logger.info(text.strip())
                        logger.info("="*50 + "\n")

                    except Exception as e:
                        logger.error(f"[PIPELINE] ❌ PARSING ERROR: {str(e)} | Raw Data: {data}")
                else:
                    logger.error(f"[PIPELINE] ❌ ENGINE ERROR: {response.status_code} - {response.text}")
                    
            except Exception as e:
                logger.error(f"[PIPELINE] ❌ NETWORK ERROR: {str(e)}")
            
            gpu_queue.task_done()

@app.on_event("startup")
async def startup():
    asyncio.create_task(process_gpu_queue())

# ==========================================
# API ROUTER
# ==========================================
@app.post("/agent/research")
async def research(query: str):
    logger.info(f"Received API request for research agent: {query}")
    prompt = f"Explain this concept thoroughly: {query}"
    
    # Push to the async queue for non-blocking execution
    await gpu_queue.put({"prompt": prompt})
    return {"status": "queued", "message": "Task dispatched to remote inference engine"}

if __name__ == "__main__":
    logger.info("Starting Disaggregated Master Node on port 8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
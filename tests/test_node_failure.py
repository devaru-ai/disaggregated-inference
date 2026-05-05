import requests
import time

MASTER_URL = "http://127.0.0.1:8080"

def test_worker_failure():
    print("========================================")
    print("INITIATING FAULT INJECTION: WORKER CRASH")
    print("========================================")
    
    print("Sending prompt to Master Node...")
    try:
        response = requests.post(
            f"{MASTER_URL}/generate", 
            params={"prompt": "Explain the theory of relativity."},
            timeout=5.0
        )
        print(f"Master Response Code: {response.status_code}")
        print(f"Master Payload: {response.json()}")
        
        if response.status_code == 502:
            print("[SUCCESS] Master successfully caught the dead worker and returned a 502 instead of crashing.")
        elif response.status_code == 200:
            print("[INFO] Worker is alive and processed the request.")
            
    except Exception as e:
        print(f"[FAILED] Master node crashed or timed out: {e}")

if __name__ == "__main__":
    test_worker_failure()
import torch
import time

def profile_oom():
    print("========================================")
    print("Starting VRAM OOM Profiler on Master GPU")
    print("========================================")
    
    if not torch.cuda.is_available():
        print("CRITICAL ERROR: CUDA not detected.")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    tensors = []
    chunk_size_gb = 1  # Allocate in 1GB chunks
    elements_per_gb = (1024 ** 3) // 4  # float32 = 4 bytes
    
    try:
        for i in range(1, 100):
            # Allocate 1GB tensor on GPU
            tensors.append(torch.randn(elements_per_gb, device='cuda', dtype=torch.float32))
            print(f"[SUCCESS] Allocated {i} GB...")
            time.sleep(0.2) # Slight delay to let memory controllers catch up
    except RuntimeError as e:
        print("\n========================================")
        print(f"[OOM REACHED] System crashed at {i - 1} GB allocated.")
        print(f"Trace: {e}")
        print("========================================")
        print("ACTION REQUIRED: Set your backpressure threshold to 80% of this limit.")

if __name__ == "__main__":
    profile_oom()
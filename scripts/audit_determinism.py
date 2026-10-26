import torch
import numpy as np
import random
import os

print("=== DETERMINISM AUDIT ===")

# Check PyTorch deterministic settings
print("=== PYTORCH SETTINGS ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")

# Check deterministic settings
print(f"\ncudnn.deterministic: {torch.backends.cudnn.deterministic}")
print(f"cudnn.benchmark: {torch.backends.cudnn.benchmark}")

# Check if seed is set in training script
train_script = r"D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal.py"
with open(train_script, "r") as f:
    content = f.read()
    if "set_seed" in content:
        print("[OK] set_seed function called")
    if "torch.manual_seed" in content:
        print("[OK] torch.manual_seed called")
    if "torch.cuda.manual_seed_all" in content:
        print("[OK] torch.cuda.manual_seed_all called")
    if "np.random.seed" in content:
        print("[OK] np.random.seed called")
    if "random.seed" in content:
        print("[OK] random.seed called")

# Check CUDA deterministic settings
print("\n=== CUDA DETERMINISM ===")
print("Note: CUDA operations may not be fully deterministic even with seeds set.")
print("For full reproducibility, set:")
print("  torch.backends.cudnn.deterministic = True")
print("  torch.backends.cudnn.benchmark = False")
print("  os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'")
print("  torch.use_deterministic_algorithms(True)")

# Check if training script sets these
train_script = r"D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal.py"
with open(train_script, "r") as f:
    content = f.read()
    if "cudnn.deterministic" in content:
        print("[OK] cudnn.deterministic set in training script")
    else:
        print("[WARN] cudnn.deterministic NOT set in training script")
    if "cudnn.benchmark" in content:
        print("[OK] cudnn.benchmark set in training script")
    else:
        print("[WARN] cudnn.benchmark NOT set in training script")
    if "CUBLAS_WORKSPACE_CONFIG" in content:
        print("[OK] CUBLAS_WORKSPACE_CONFIG set")
    else:
        print("[WARN] CUBLAS_WORKSPACE_CONFIG NOT set")
    if "use_deterministic_algorithms" in content:
        print("[OK] use_deterministic_algorithms set")
    else:
        print("[WARN] use_deterministic_algorithms NOT set")

# Check DataLoader workers
print("\n=== DATALOADER WORKERS ===")
print("Training script uses --workers argument (default 0)")
print("With workers=0, data loading is in main process (more deterministic)")
print("With workers>0, multiprocessing introduces non-determinism")

# Check environment
print("\n=== ENVIRONMENT ===")
print(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', 'not set')}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

print("\n=== DETERMINISM SUMMARY ===")
print("Seed is set to 42 via --seed argument")
print("set_seed() sets: random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all")
print("WARNING: Full CUDA determinism requires additional settings not currently in training script")
print("GPU execution may produce slightly different results across runs due to:")
print("  - cuDNN non-deterministic algorithms")
print("  - Floating-point non-associativity in parallel operations")
print("  - Atomic operations in CUDA kernels")
print("For strict reproducibility, consider CPU-only training or full deterministic settings")
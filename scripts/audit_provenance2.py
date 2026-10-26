import pandas as pd
from pathlib import Path

# Check cache generation script
cache_script = r"D:\AI-based-Deepfake-Detection-system-main\scripts\cache_celebdf_resume.py"
print("=== CACHE GENERATION SCRIPT ===")
with open(cache_script, "r") as f:
    content = f.read()
    # Check key parameters
    if "T, SIZE, FEATURE_SIZE = 20, 224, 1280" in content:
        print("[OK] T=20, SIZE=224, FEATURE_SIZE=1280")
    if "EfficientNet_B0_Weights.DEFAULT" in content:
        print("[OK] EfficientNet-B0 with default weights")
    if "cv2.COLOR_BGR2RGB" in content:
        print("[OK] BGR to RGB conversion")
    if "cv2.resize" in content and "(SIZE, SIZE)" in content:
        print("[OK] Resize to 224x224")
    if "frame.astype(np.float32) / 255.0" in content:
        print("[OK] Normalization to [0, 1]")
    if "mean = torch.tensor([0.485, 0.456, 0.406]" in content:
        print("[OK] ImageNet mean")
    if "std = torch.tensor([0.229, 0.224, 0.225]" in content:
        print("[OK] ImageNet std")
    if "frames = torch.from_numpy(frames).permute(0, 3, 1, 2)" in content:
        print("[OK] Permute to (T, C, H, W)")
    if "normalized = (frames_tensor - mean) / std" in content:
        print("[OK] ImageNet normalization")
    if "backbone(batch)" in content:
        print("[OK] EfficientNet-B0 backbone feature extraction")
    if "np.save(temp, features)" in content and "temp.replace(output)" in content:
        print("[OK] Atomic save with .tmp extension")

# Check inference preprocessing
inference_script = r"D:\AI-based-Deepfake-Detection-system-main\backend\app\ml\torch_inference.py"
print("\n=== INFERENCE PREPROCESSING ===")
with open(inference_script, "r") as f:
    content = f.read()
    if "cv2.resize" in content and "(settings.image_size, settings.image_size)" in content:
        print("[OK] Resize to image_size x image_size")
    if "cv2.INTER_LINEAR" in content:
        print("[OK] INTER_LINEAR interpolation")
    if "faces_to_tensor" in content:
        print("[OK] faces_to_tensor function")
    if "select_evenly_spaced_frames" in content:
        print("[OK] select_evenly_spaced_frames")
    if "mean = torch.tensor([0.485, 0.456, 0.406]" in content:
        print("[OK] ImageNet mean in inference")
    if "std = torch.tensor([0.229, 0.224, 0.225]" in content:
        print("[OK] ImageNet std in inference")
    if "normalized = (frames - mean) / std" in content:
        print("[OK] ImageNet normalization in inference")
    if "backbone(normalized" in content:
        print("[OK] EfficientNet-B0 backbone in inference")

# Check temporal sampling
sampling_script = r"D:\AI-based-Deepfake-Detection-system-main\backend\app\ml\temporal_sampling.py"
print("\n=== TEMPORAL SAMPLING ===")
with open(sampling_script, "r") as f:
    content = f.read()
    if "select_evenly_spaced_frames" in content:
        print("[OK] select_evenly_spaced_frames function exists")
    if "np.linspace" in content:
        print("[OK] Uses np.linspace for even spacing")

# Check training script uses correct cache dir
train_script = r"D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal.py"
print("\n=== TRAINING SCRIPT CACHE USAGE ===")
with open(train_script, "r") as f:
    content = f.read()
    if "args.cache_dir" in content:
        print("[OK] Uses args.cache_dir argument")
    if "CachedFeatureDataset" in content:
        print("[OK] Uses CachedFeatureDataset")
    if "row[\"feature_path\"]" in content:
        print("[OK] Reads feature_path from index")
    if "split" in content and "usable[usable[\"split\"] == \"train\"]" in content:
        print("[OK] Uses split column from index for train/val split")

print("\n=== PROVENANCE NOTE ===")
print("The cache files (.npy) contain only the extracted features (20, 1280).")
print("There is no embedded source-video hash, frame indices, or preprocessing metadata in the .npy files.")
print("Feature-to-video provenance is established via the feature_index.csv which maps")
print("relative_path -> feature_path, and the manifest maps relative_path -> video_path.")
print("The cache generation script (cache_celebdf_resume.py) uses the manifest's video_path")
print("to load videos, but the resulting .npy files do not contain provenance metadata.")
print("Provenance is traceable via the index/manifest chain but not embedded in .npy files.")
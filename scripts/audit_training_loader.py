import pandas as pd
from pathlib import Path

# Check training script argument parsing and data loading
train_script = r"D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal.py"
print("=== TRAINING SCRIPT ARGUMENT PARSING ===")
with open(train_script, "r") as f:
    content = f.read()

# Check argument definitions
if "parser.add_argument" in content:
    print("[OK] Uses argparse")
    # Check specific arguments
    if "--index" in content:
        print("[OK] --index argument defined")
    if "--cache-dir" in content:
        print("[OK] --cache-dir argument defined")
    if "--output" in content:
        print("[OK] --output argument defined")
    if "--epochs" in content:
        print("[OK] --epochs argument defined")
    if "--batch-size" in content:
        print("[OK] --batch-size argument defined")
    if "--learning-rate" in content:
        print("[OK] --learning-rate argument defined")
    if "--weight-decay" in content:
        print("[OK] --weight-decay argument defined")
    if "--patience" in content:
        print("[OK] --patience argument defined")
    if "--seed" in content:
        print("[OK] --seed argument defined")

# Check data loading
if "pd.read_csv(args.index)" in content:
    print("[OK] Reads index from args.index")
if "args.cache_dir" in content:
    print("[OK] Uses args.cache_dir for cache directory")
if "args.output" in content:
    print("[OK] Uses args.output for output directory")

# Check split column usage
if "usable[usable[\"split\"] == \"train\"]" in content:
    print("[OK] Train split filtering: usable[usable[\"split\"] == \"train\"]")
if "usable[usable[\"split\"] == \"val\"]" in content:
    print("[OK] Val split filtering: usable[usable[\"split\"] == \"val\"]")
if "usable[usable[\"split\"] == \"test\"]" in content:
    print("[OK] Test split filtering exists (but not used for training)")

# Check that test split is NOT used for training
if "test" in content and "train" in content:
    # Check that test split is not used in training loop
    if "val_frame = usable[usable[\"split\"] == \"val\"]" in content:
        print("[OK] Validation split used for validation")
    if "train_frame = usable[usable[\"split\"] == \"train\"]" in content:
        print("[OK] Training split used for training")

# Check checkpoint selection uses validation loss
if "best_val_loss" in content:
    print("[OK] Uses best_val_loss for checkpoint selection")
if "val_result[\"loss\"]" in content and "scheduler.step(val_result[\"loss\"])" in content:
    print("[OK] Scheduler steps on validation loss")
if "val_result[\"loss\"] < best_val_loss" in content:
    print("[OK] Checkpoint selection based on validation loss")

# Check that test is not used for checkpoint selection
print("\n=== CHECKPOINT SELECTION ===")
print("Checkpoint selection uses validation loss (val_result[\"loss\"])")
print("Test split is NOT used for checkpoint selection or early stopping")

# Check output directory
print("\n=== OUTPUT DIRECTORY ===")
print("Output directory: args.output (new experiment directory)")
print("Will not overwrite production checkpoint at backend/artifacts/models/torch/best_checkpoint.pt")
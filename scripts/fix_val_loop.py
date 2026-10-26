with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2_new.py', 'r') as f:
    content = f.read()

# Fix the validation loop indentation
old = '''print("\\n=== THRESHOLD SELECTION ON VALIDATION ===")
    val_labels, val_logits = [], []
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
                val_output = model(features)
                val_logits.extend(val_output.detach().cpu().tolist())
            val_labels.extend(targets.detach().cpu().tolist())'''

new = '''print("\\n=== THRESHOLD SELECTION ON VALIDATION ===")
    val_labels, val_logits = [], []
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            val_output = model(features)
            val_logits.extend(val_output.detach().cpu().tolist())
            val_labels.extend(targets.detach().cpu().tolist())'''

content = content.replace(old, new)

with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2_new.py', 'w') as f:
    f.write(content)

print('Fixed validation loop indentation')
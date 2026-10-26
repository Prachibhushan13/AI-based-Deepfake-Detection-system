with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'r') as f:
    lines = f.readlines()

# Fix line 511 (index 510) - change from 8 spaces to 4 spaces
lines[510] = '    print(f"\\n=== TEST EVALUATION (best checkpoint, threshold={best_threshold:.2f}) ===")\n'
lines[511] = '    test_labels, test_logits = [], []\n'
lines[512] = '    test_loss = 0.0\n'
lines[513] = '    with torch.no_grad():\n'

with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'w') as f:
    f.writelines(lines)

print('Fixed lines 511-514')
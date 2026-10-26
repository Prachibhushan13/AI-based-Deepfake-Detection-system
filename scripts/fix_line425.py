with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'r') as f:
    lines = f.readlines()

# Fix line 425 (index 424) - change from 8 spaces to 12 spaces
lines[424] = '            "macro_f1": float(test_result["macro_f1"]),\n'

with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'w') as f:
    f.writelines(lines)

print('Fixed line 425')
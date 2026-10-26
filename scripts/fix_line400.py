with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'r') as f:
    lines = f.readlines()

# Fix line 400 (index 399) - change from 8 spaces to 4 spaces
lines[399] = '    test_loss = 0.0\n'

with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'w') as f:
    f.writelines(lines)

print('Fixed line 400')
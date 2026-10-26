with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'r') as f:
    lines = f.readlines()

# Fix line 401 (index 400) - change from 8 spaces to 4 spaces
lines[400] = '    with torch.no_grad():\n'

with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'w') as f:
    f.writelines(lines)

print('Fixed line 401')
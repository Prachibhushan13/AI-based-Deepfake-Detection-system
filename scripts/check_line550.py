with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'rb') as f:
    content = f.read()
lines = content.split(b'\n')
line_550 = lines[549]
print('Line 550 raw:', line_550)
print('Line 550 hex:', line_550.hex())
print('Line 550 length:', len(line_550))
print('Starts with spaces:', line_550.startswith(b' '))
print('First 20 bytes:', line_550[:20].hex())
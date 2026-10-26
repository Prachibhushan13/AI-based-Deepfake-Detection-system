with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'rb') as f:
    content = f.read()

# Replace Windows-style line endings with Unix-style
content = content.replace(b'\r\n', b'\n')

with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'wb') as f:
    f.write(content)

print('Fixed CRLF')
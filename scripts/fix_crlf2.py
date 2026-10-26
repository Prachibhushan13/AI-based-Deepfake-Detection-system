import os
import sys

# Read the current script
with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'r') as f:
    content = f.read()

# The script has indentation issues. Let's rewrite it completely.
# I'll create a new version with correct indentation.

# Actually, let's just try to fix the specific issue by removing the \r characters
# and ensuring consistent indentation.

# Replace \r\n with \n
content = content.replace(b'\r\n', b'\n').replace(b'\r', b'')

# Write back
with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'wb') as f:
    f.write(content)

print('Fixed CRLF and any other line ending issues')
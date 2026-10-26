import os
import subprocess

# Check WSL availability and paths
print('=== WSL Check ===')
try:
    result = subprocess.run(['wsl', '--list'], capture_output=True, text=True, timeout=10)
    print('WSL distros:', result.stdout)
except Exception as e:
    print('WSL error:', e)

# Check common WSL mount points
wsl_paths = [
    '/mnt/d/Nimisha Deepfake Project/AI-based-Deepfake-Detection-system/datasets/Celeb-DF-v2',
    '/home',
    '/mnt/d',
]
print()
print('=== WSL Path Check (via wsl command) ===')
for p in wsl_paths:
    try:
        result = subprocess.run(['wsl', 'test', '-e', p], capture_output=True, timeout=5)
        exists = result.returncode == 0
        print('  {} -> {}'.format(p, 'EXISTS' if exists else 'NOT FOUND'))
    except Exception as e:
        print('  {} -> ERROR: {}'.format(p, e))

# Check if there's a datasets folder in the project
print()
print('=== Project datasets folder ===')
project_datasets = r'D:\AI-based-Deepfake-Detection-system-main\datasets'
if os.path.exists(project_datasets):
    print('EXISTS: {}'.format(project_datasets))
    for root, dirs, files in os.walk(project_datasets):
        level = root.replace(project_datasets, '').count(os.sep)
        indent = ' ' * 2 * level
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 2 * (level + 1)
        for f in files[:10]:
            print('{}{}'.format(subindent, f))
        if len(files) > 10:
            print('{}... and {} more files'.format(subindent, len(files)-10))
else:
    print('NOT FOUND: {}'.format(project_datasets))
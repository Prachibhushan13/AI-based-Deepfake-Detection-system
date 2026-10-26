with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'r') as f:
    content = f.read()

# Fix the indentation issue
content = content.replace(
    '    # Test evaluation at best threshold\n        print(f"\n=== TEST EVALUATION (best checkpoint, threshold={best_threshold:.2f}) ===")',
    '    # Test evaluation at best threshold\n    print(f"\n=== TEST EVALUATION (best checkpoint, threshold={best_threshold:.2f}) ===")'
)

with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'w') as f:
    f.write(content)

print('Fixed')
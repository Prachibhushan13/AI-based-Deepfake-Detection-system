with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'r') as f:
    lines = f.readlines()

# Fix test_metrics dict entries - should be 12 spaces (8 for dict + 4 for entry)
# Lines 418-428 (indices 417-427)
lines[417] = '        "test_loss": float(test_result["loss"].item()) if hasattr(test_result["loss"], "item") else float(test_result["loss"]),\n'
lines[418] = '        "accuracy": float(test_result["accuracy"]),\n'
lines[419] = '        "balanced_accuracy": float(test_result["balanced_accuracy"]),\n'
lines[420] = '        "precision_real": float(test_result["precision_real"]),\n'
lines[421] = '        "recall_real": float(test_result["recall_real"]),\n'
lines[421] = '        "f1_real": float(test_result["f1_real"]),\n'
lines[422] = '        "precision_fake": float(test_result["precision_fake"]),\n'
lines[422] = '        "recall_fake": float(test_result["recall_fake"]),\n'
lines[423] = '        "f1_fake": float(test_result["f1_fake"]),\n'
lines[424] = '        "macro_f1": float(test_result["macro_f1"]),\n'
lines[425] = '        "roc_auc": float(test_result["roc_auc"]),\n'
lines[426] = '        "confusion_matrix": test_result["confusion_matrix"],\n'
lines[427] = '        "threshold_used": float(best_threshold),\n'

with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'w') as f:
    f.writelines(lines)

print('Fixed test_metrics dict indentation')
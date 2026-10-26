with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2_new.py', 'r') as f:
    content = f.read()

# Find and replace the test_metrics dict
old = 'test_metrics = {\n'
old += '        "test_loss": float(test_result["loss"]),\n'
old += '        "accuracy": float(test_result["accuracy"]),\n'
old += '        "balanced_accuracy": float(test_result["balanced_accuracy"]),\n'
old += '        "precision_real": float(test_result["precision_real"]),\n'
old += '        "recall_real": float(test_result["recall_real"]),\n'
old += '        "f1_real": float(test_result["f1_real"]),\n'
old += '        "precision_fake": float(test_result["precision_fake"]),\n'
old += '        "recall_fake": float(test_result["recall_fake"]),\n'
old += '        "f1_fake": float(test_result["f1_fake"]),\n'
old += '        "macro_f1": float(test_result["macro_f1"]),\n'
old += '        "roc_auc": float(test_result["roc_auc"]),\n'
old += '        "confusion_matrix": test_result["confusion_matrix"],\n'
old += '        "threshold_used": float(best_threshold),\n'
old += '    }'

new = 'test_metrics = {\n'
new += '        "test_loss": float(test_result["loss"].item()) if hasattr(test_result["loss"], "item") else float(test_result["loss"]),\n'
new += '        "accuracy": float(test_result["accuracy"]),\n'
new += '        "balanced_accuracy": float(test_result["balanced_accuracy"]),\n'
new += '        "precision_real": float(test_result["precision_real"]),\n'
new += '        "recall_real": float(test_result["recall_real"]),\n'
new += '        "f1_real": float(test_result["f1_real"]),\n'
new += '        "precision_fake": float(test_result["precision_fake"]),\n'
new += '        "recall_fake": float(test_result["recall_fake"]),\n'
new += '        "f1_fake": float(test_result["f1_fake"]),\n'
new += '        "macro_f1": float(test_result["macro_f1"]),\n'
new += '        "roc_auc": float(test_result["roc_auc"]),\n'
new += '        "confusion_matrix": test_result["confusion_matrix"],\n'
new += '        "threshold_used": float(best_threshold),\n'
new += '    }'

content = content.replace(
    'test_metrics = {\n'
    '        "test_loss": float(test_result["loss"]),\n'
    '        "accuracy": float(test_result["accuracy"]),\n'
    '        "balanced_accuracy": float(test_result["balanced_accuracy"]),\n'
    '        "precision_real": float(test_result["precision_real"]),\n'
    '        "recall_real": float(test_result["recall_real"]),\n'
    '        "f1_real": float(test_result["f1_real"]),\n'
    '        "precision_fake": float(test_result["precision_fake"]),\n'
    '        "recall_fake": float(test_result["recall_fake"]),\n'
    '        "f1_fake": float(test_result["f1_fake"]),\n'
    '        "macro_f1": float(test_result["macro_f1"]),\n'
    '        "roc_auc": float(test_result["roc_auc"]),\n'
    '        "confusion_matrix": test_result["confusion_matrix"],\n'
    '        "threshold_used": float(best_threshold),\n'
    '    }',
    new)

with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2_new.py', 'w') as f:
    f.write(content)

print('Fixed test_metrics dict')
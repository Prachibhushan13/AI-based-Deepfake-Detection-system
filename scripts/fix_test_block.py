with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'r') as f:
    lines = f.readlines()

# Fix lines 413-420 (indices 412-420) - change from 8 spaces to 4 spaces
lines[412] = '    test_probs = 1.0 / (1.0 + np.exp(-np.asarray(test_logits)))\n'
lines[413] = '    test_labels_np = np.asarray(test_labels, dtype=np.int64)\n'
lines[413] = '    test_preds = (test_probs >= best_threshold).astype(int)\n'
lines[414] = '\n'
lines[415] = '    test_result = compute_metrics(test_labels, test_logits, best_threshold)\n'
lines[416] = '    test_result["loss"] = test_loss / len(test_loader.dataset)\n'
lines[417] = '    test_metrics = {\n'
lines[417] = '        "test_loss": float(test_result["loss"].item()) if hasattr(test_result["loss"], "item") else float(test_result["loss"]),\n'
lines[418] = '        "accuracy": float(test_result["accuracy"]),\n'
lines[418] = '        "balanced_accuracy": float(test_result["balanced_accuracy"]),\n'
lines[419] = '        "precision_real": float(test_result["precision_real"]),\n'
lines[419] = '        "recall_real": float(test_result["recall_real"]),\n'
lines[420] = '        "f1_real": float(test_result["f1_real"]),\n'
lines[421] = '        "precision_fake": float(test_result["precision_fake"]),\n'
lines[422] = '        "recall_fake": float(test_result["recall_fake"]),\n'
lines[423] = '        "f1_fake": float(test_result["f1_fake"]),\n'
lines[424] = '        "macro_f1": float(test_result["macro_f1"]),\n'
lines[425] = '        "roc_auc": float(test_result["roc_auc"]),\n'
lines[426] = '        "confusion_matrix": test_result["confusion_matrix"],\n'
lines[427] = '        "threshold_used": float(best_threshold),\n'
lines[428] = '    }\n'
lines[429] = '    (output / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))\n'

with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'w') as f:
    f.writelines(lines)

print('Fixed test evaluation block')
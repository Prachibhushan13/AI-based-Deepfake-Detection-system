with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'r') as f:
    lines = f.readlines()

# Fix lines 413-420 (indices 412-419) - change from 8 spaces to 4 spaces
lines[412] = '    test_probs = 1.0 / (1.0 + np.exp(-np.asarray(test_logits)))\n'
lines[413] = '    test_labels_np = np.asarray(test_labels, dtype=np.int64)\n'
lines[414] = '    test_preds = (test_probs >= best_threshold).astype(int)\n'
lines[415] = '\n'
lines[416] = '    test_result = compute_metrics(test_labels, test_logits, best_threshold)\n'
lines[417] = '    test_result["loss"] = test_loss / len(test_loader.dataset)\n'
lines[418] = '    test_metrics = {\n'
lines[419] = '        "test_loss": float(test_result["loss"]),\n'

with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'w') as f:
    f.writelines(lines)

print('Fixed lines 413-419')
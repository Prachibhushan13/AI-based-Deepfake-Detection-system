with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'r') as f:
    lines = f.readlines()

# Fix lines 511-540 (0-indexed: 510-540)
# Ensure consistent 4-space indentation for the test evaluation block
# Lines 510-540 (0-indexed 509-539)

# Line 510 (index 510): comment - should be 4 spaces
# Line 511 (index 511): print - should be 4 spaces
# Line 512 (index 512): test_labels - 4 spaces
# Line 512: test_loss = 0.0 - 4 spaces
# Line 513: with torch.no_grad(): - 4 spaces
# Lines 514-524: inside with block - 8 spaces
# Line 525: empty line - 0 spaces
# Line 526: test_probs = ... - 4 spaces
# Lines 526-540: 4 spaces

# Fix line 511 (index 510) - print statement
lines[510] = '    print(f"\\n=== TEST EVALUATION (best checkpoint, threshold={best_threshold:.2f}) ===")\n'
# Line 512 (index 511)
lines[511] = '    test_labels, test_logits = [], []\n'
# Line 513 (index 512)
lines[512] = '    test_loss = 0.0\n'
# Line 514 (index 513)
lines[513] = '    with torch.no_grad():\n'
# Lines 514-524 (indices 514-524) should be 8 spaces - already correct
# Line 525 (index 524) - empty line, should be just newline
lines[524] = '\n'
# Line 526 (index 525) - test_probs = ... should be 4 spaces
lines[525] = '    test_probs = 1.0 / (1.0 + np.exp(-np.asarray(test_logits)))\n'
# Line 527 (index 526)
lines[526] = '    test_labels_np = np.asarray(test_labels, dtype=np.int64)\n'
# Line 528 (index 527)
lines[527] = '    test_preds = (test_probs >= best_threshold).astype(int)\n'
# Line 529 (index 528) - empty line
lines[528] = '\n'
# Line 530 (index 529)
lines[529] = '    test_result = compute_metrics(test_labels, test_logits, best_threshold)\n'
# Line 531 (index 530)
lines[530] = '    test_result["loss"] = test_loss / len(test_loader.dataset)\n'
# Line 531 (index 531) - test_metrics dict
lines[531] = '    test_metrics = {\n'
# Lines 532-540 (indices 532-540) - dict entries, 8 spaces
# Line 541 (index 541) - closing brace and write
lines[541] = '    }\n'
# Line 542 (index 542)
lines[542] = '    (output / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))\n'
# Line 543 (index 543)
lines[543] = '    print(f"Test: loss={test_result[\'loss\']:.4f} acc={test_result[\'accuracy\']:.4f} "\n'
# Line 544 (index 544)
lines[544] = '          f"bal_acc={test_result[\'balanced_accuracy\']:.4f} macro_f1={test_result[\'macro_f1\']:.4f} "\n'
# Line 545 (index 545)
lines[545] = '          f"prec_real={test_result[\'precision_real\']:.4f} rec_real={test_result[\'recall_real\']:.4f} "\n'
# Line 546 (index 546)
lines[546] = '          f"prec_fake={test_result[\'precision_fake\']:.4f} rec_fake={test_result[\'recall_fake\']:.4f} "\n'
# Line 547 (index 547)
lines[547] = '          f"macro_f1={test_result[\'macro_f1\']:.4f} auc={test_result[\'roc_auc\']} "\n'
# Line 548 (index 548)
lines[548] = '          f"cm={test_result[\'confusion_matrix\']}")\n'

with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'w') as f:
    f.writelines(lines)

print('Fixed lines 511-548')
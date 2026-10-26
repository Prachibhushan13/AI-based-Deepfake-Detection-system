with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'r') as f:
    lines = f.readlines()

# Fix lines 550-553 (0-indexed: 549-552) - should be 10 spaces, not 14
lines[549] = '          f"prec_real={test_result[\'precision_real\']:.4f} rec_real={test_result[\'recall_real\']:.4f} "\n'
lines[550] = '          f"prec_fake={test_result[\'precision_fake\']:.4f} rec_fake={test_result[\'recall_fake\']:.4f} "\n'
lines[551] = '          f"macro_f1={test_result[\'macro_f1\']:.4f} auc={test_result[\'roc_auc\']} "\n'
lines[552] = '          f"cm={test_result[\'confusion_matrix\']}")\n'

with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'w') as f:
    f.writelines(lines)

print('Fixed lines 550-553')
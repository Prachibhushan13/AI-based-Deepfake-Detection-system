import json
with open(r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\threshold_analysis_20261026\corrected_threshold_analysis.json') as f:
    data = json.load(f)
print('Keys:', list(data.keys()))
print()
best = data['best_balanced_accuracy']
print('Best balanced accuracy:')
print('  threshold={:.2f}, bal_acc={:.4f}'.format(best['threshold'], best['balanced_accuracy']))
print('  REAL_recall={:.4f}, FAKE_recall={:.4f}'.format(best['real_recall'], best['fake_recall']))
print('  REAL_precision={:.4f}, FAKE_precision={:.4f}'.format(best['real_precision'], best['fake_precision']))
print()
print('Correction note:', data['correction_note'])
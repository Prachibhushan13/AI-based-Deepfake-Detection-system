import pandas as pd
import numpy as np

eval_dir = r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\sdfvd_evaluation'
pred_df = pd.read_csv(eval_dir + '/predictions.csv')

# All successful
success = pred_df[pred_df['status'] == 'success']
print('Total successful:', len(success))
print()

# Group by true label
real_probs = success[success['true_label'] == 0]['fake_probability']
fake_probs = success[success['true_label'] == 1]['fake_probability']

print('=== REAL (label 0) fake probabilities ===')
print('Count:', len(real_probs))
print('Mean:', real_probs.mean())
print('Median:', real_probs.median())
print('Std:', real_probs.std())
print('Min:', real_probs.min())
print('Max:', real_probs.max())
print('25th/75th percentile:', real_probs.quantile(0.25), '/', real_probs.quantile(0.75))
print()

print('=== FAKE (label 1) fake probabilities ===')
print('Count:', len(fake_probs))
print('Mean:', fake_probs.mean())
print('Median:', fake_probs.median())
print('Std:', fake_probs.std())
print('Min:', fake_probs.min())
print('Max:', fake_probs.max())
print('25th/75th percentile:', fake_probs.quantile(0.25), '/', fake_probs.quantile(0.75))
print()

# Overlap
print('=== OVERLAP ===')
print('REAL max:', real_probs.max())
print('FAKE min:', fake_probs.min())
print('REAL probs > 0.5:', (real_probs > 0.5).sum(), '/', len(real_probs))
print('FAKE probs < 0.5:', (fake_probs < 0.5).sum(), '/', len(fake_probs))
print()

# Misclassified videos
success['correct'] = success['true_label'] == success['pred_label']
misclassified = success[~success['correct']].copy()
misclassified['confidence'] = abs(misclassified['fake_probability'] - 0.53296262)
misclassified = misclassified.sort_values('confidence', ascending=False)

print('=== TOP 10 MOST CONFIDENTLY MISCLASSIFIED ===')
for _, row in misclassified.head(10).iterrows():
    true_label = 'REAL' if row['true_label'] == 0 else 'FAKE'
    pred_label = 'REAL' if row['pred_label'] == 0 else 'FAKE'
    print('{}: true={}, pred={}, prob={:.4f}, conf={:.4f}'.format(
        row['filename'], true_label, pred_label, row['fake_probability'], row['confidence']))
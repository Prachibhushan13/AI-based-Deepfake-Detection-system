import json
import pandas as pd
from pathlib import Path

eval_dir = Path(r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\sdfvd_evaluation')

# Read predictions.csv
pred_df = pd.read_csv(eval_dir / 'predictions.csv')
print('=== PREDICTIONS.CSV ===')
print('Shape:', pred_df.shape)
print('Columns:', list(pred_df.columns))
print('Status counts:', pred_df['status'].value_counts().to_dict())
print()
print('First 10 rows:')
print(pred_df.head(10).to_string())
print()

# Read metrics.json
with open(eval_dir / 'metrics.json') as f:
    metrics = json.load(f)
print('=== METRICS.JSON ===')
print(json.dumps(metrics, indent=2))
print()

# Read run_metadata.json
with open(eval_dir / 'run_metadata.json') as f:
    metadata = json.load(f)
print('=== RUN_METADATA.JSON ===')
print(json.dumps(metadata, indent=2))
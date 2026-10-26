"""
Create identity-disjoint feature index matching the identity-disjoint manifest.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load identity-disjoint manifest
manifest_path = r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_identity_disjoint_20261023\celebdf_manifest.csv'
manifest = pd.read_csv(manifest_path)

# Load existing feature index
index_path = r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_retrain_20261023\features_index.csv'
old_index = pd.read_csv(index_path)

# Merge manifest with old index to get feature paths and status
# The manifest has relative_path, folder, label, split
# The old index has relative_path, feature_path, label, split, status, error
new_index = manifest.merge(
    old_index[['relative_path', 'feature_path', 'status', 'error']],
    on='relative_path',
    how='left'
)

print('New index shape:', new_index.shape)
print('Split distribution:')
print(new_index['split'].value_counts())
print()
print('Label by split:')
print(new_index.groupby('split')['label'].value_counts().unstack(fill_value=0))
print()
print('Status distribution:')
print(new_index['status'].value_counts())
print()

# Check for missing feature paths
missing = new_index[new_index['feature_path'].isna()]
print(f'Missing feature paths: {len(missing)}')
if len(missing) > 0:
    print(missing[['relative_path', 'folder', 'label', 'split']].head())

# Save new index
output_dir = Path(r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\identity_audit_v2')
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / 'features_index.csv'
new_index.to_csv(output_path, index=False)
print(f'\nSaved new index to {output_path}')

# Verify feature files exist
cache_dir = Path(r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_retrain_20261023\celebdf_features')
missing_files = 0
for _, row in new_index.iterrows():
    fpath = Path(row['feature_path'])
    if not fpath.exists():
        fpath = cache_dir / Path(row['feature_path']).name
    if not fpath.exists():
        missing_files += 1
        print(f'MISSING: {row["feature_path"]}')

print(f'Missing feature files: {missing_files}')

# Save split counts
print('\n=== SPLIT COUNTS ===')
print(new_index.groupby(['split', 'label']).size().unstack(fill_value=0))
print()
print(new_index.groupby(['split', 'folder']).size().unstack(fill_value=0))
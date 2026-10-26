import pandas as pd
import numpy as np
from pathlib import Path

# Load manifest
manifest_path = r'D:\AI-based-Deepfake-Detection-system-main\datasets\celebdf_manifest.csv'
manifest = pd.read_csv(manifest_path)
print('=== MANIFEST ===')
print('Shape:', manifest.shape)
print('Columns:', list(manifest.columns))
print()
print('Label distribution:')
print(manifest['label'].value_counts())
print()
print('Split distribution:')
print(manifest['split'].value_counts())
print()
print('Label by split:')
print(manifest.groupby('split')['label'].value_counts())
print()
print('Folder distribution:')
print(manifest['folder'].value_counts())
print()

# Load feature index
index_path = r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_retrain_20261023\features_index.csv'
index = pd.read_csv(index_path)
print('=== FEATURE INDEX ===')
print('Shape:', index.shape)
print('Columns:', list(index.columns))
print()
print('Status distribution:')
print(index['status'].value_counts())
print()
print('Label distribution:')
print(index['label'].value_counts())
print()
print('Split distribution:')
print(index['split'].value_counts())
print()
print('Label by split:')
print(index.groupby('split')['label'].value_counts())
print()

# Check for duplicates
print('=== DUPLICATE CHECKS ===')
print('Duplicate relative_path in manifest:', manifest['relative_path'].duplicated().sum())
print('Duplicate relative_path in index:', index['relative_path'].duplicated().sum())
print('Duplicate feature_path in index:', index['feature_path'].duplicated().sum())
print()

# Check feature files
cache_dir = Path(r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_retrain_20261023\celebdf_features')
print('=== FEATURE FILES ===')
print('Cache dir exists:', cache_dir.exists())
npy_files = list(cache_dir.glob('*.npy'))
print('Number of .npy files:', len(npy_files))

# Check a sample of feature files
print()
print('=== SAMPLE FEATURE VALIDATION ===')
sample = index[index['status'] == 'ok'].sample(n=min(10, len(index)), random_state=42)
for _, row in sample.iterrows():
    fpath = Path(row['feature_path'])
    if not fpath.exists():
        fpath = cache_dir / Path(row['feature_path']).name
    if fpath.exists():
        arr = np.load(fpath)
        print('{}: shape={}, dtype={}, finite={}, mean={:.4f}, std={:.4f}'.format(
            fpath.name, arr.shape, arr.dtype, np.isfinite(arr).all(), arr.mean(), arr.std()))
    else:
        print('MISSING: {}'.format(row['feature_path']))

# Full validation of all ok features
print()
print('=== FULL FEATURE VALIDATION ===')
ok_index = index[index['status'] == 'ok']
missing = 0
bad_shape = 0
non_finite = 0
for _, row in ok_index.iterrows():
    fpath = Path(row['feature_path'])
    if not fpath.exists():
        fpath = cache_dir / Path(row['feature_path']).name
    if not fpath.exists():
        missing += 1
        continue
    arr = np.load(fpath)
    if arr.shape != (20, 1280):
        bad_shape += 1
    if not np.isfinite(arr).all():
        non_finite += 1
print('Missing files:', missing)
print('Bad shape:', bad_shape)
print('Non-finite values:', non_finite)
print('Total ok:', len(ok_index))

# Check label consistency between manifest and index
print()
print('=== LABEL CONSISTENCY ===')
# Merge on relative_path
merged = manifest.merge(index[['relative_path', 'label', 'split']], on='relative_path', suffixes=('_manifest', '_index'))
label_mismatch = merged[merged['label_manifest'] != merged['label_index']]
split_mismatch = merged[merged['split_manifest'] != merged['split_index']]
print('Label mismatches:', len(label_mismatch))
print('Split mismatches:', len(split_mismatch))
if len(label_mismatch) > 0:
    print(label_mismatch[['relative_path', 'label_manifest', 'label_index']].head())
if len(split_mismatch) > 0:
    print(split_mismatch[['relative_path', 'split_manifest', 'split_index']].head())

# Identity overlap check
print()
print('=== IDENTITY OVERLAP CHECK ===')
# Extract identity from folder/relative_path
# Celeb-real/id0_0001.mp4 -> id0
def extract_id(path):
    # path like "Celeb-real/id0_0001.mp4" or "Celeb-synthesis/id0_id1_0001.mp4"
    parts = path.split('/')
    if len(parts) >= 2:
        filename = parts[-1]
        # For real: id0_0001.mp4 -> id0
        # For synthesis: id0_id1_0001.mp4 -> id0_id1 (but we want source identities)
        if 'id' in filename:
            # Extract all idX patterns
            import re
            ids = re.findall(r'id\d+', filename)
            return tuple(sorted(ids))
    return None

manifest['identities'] = manifest['relative_path'].apply(extract_id)
print('Unique identity groups in manifest:', manifest['identities'].nunique())
print('Identity group sizes:')
print(manifest.groupby('identities').size().describe())

# Check overlap between splits
train_ids = set(manifest[manifest['split'] == 'train']['identities'].dropna())
val_ids = set(manifest[manifest['split'] == 'val']['identities'].dropna())
test_ids = set(manifest[manifest['split'] == 'test']['identities'].dropna())
print()
print('Train identities:', len(train_ids))
print('Val identities:', len(val_ids))
print('Test identities:', len(test_ids))
print('Train-Val overlap:', len(train_ids & val_ids))
print('Train-Test overlap:', len(train_ids & test_ids))
print('Val-Test overlap:', len(val_ids & test_ids))
import pandas as pd
import re

# Check identity-disjoint manifest
path = r'D:\AI-based-Deepfake-Detection-system-main\backend\artifacts\models\torch\celebdf_identity_disjoint_20261023\celebdf_manifest.csv'
df = pd.read_csv(path)
print('Shape:', df.shape)
print('Columns:', list(df.columns))
print()
print('Split distribution:')
print(df['split'].value_counts())
print()
print('Label by split:')
print(df.groupby('split')['label'].value_counts().unstack(fill_value=0))
print()
print('Folder by split:')
print(df.groupby('split')['folder'].value_counts().unstack(fill_value=0))
print()

# Check identity overlap in this manifest
def extract_identities(path):
    filename = path.split('/')[-1]
    if filename.startswith('id'):
        ids = re.findall(r'id(\d+)', filename)
        return [int(x) for x in ids]
    else:
        return []

df['identities'] = df['relative_path'].apply(extract_identities)

for split_name in ['train', 'val', 'test']:
    split_ids = set()
    for ids in df[df['split'] == split_name]['identities']:
        split_ids.update(ids)
    print('{} identities: {}'.format(split_name, len(split_ids)))

train_ids = set()
for ids in df[df['split'] == 'train']['identities']:
    train_ids.update(ids)
val_ids = set()
for ids in df[df['split'] == 'val']['identities']:
    val_ids.update(ids)
test_ids = set()
for ids in df[df['split'] == 'test']['identities']:
    test_ids.update(ids)

print()
print('Train-Val overlap: {}'.format(len(train_ids & val_ids)))
print('Train-Test overlap: {}'.format(len(train_ids & test_ids)))
print('Val-Test overlap: {}'.format(len(val_ids & test_ids)))

# Check YouTube-real
yt = df[df['relative_path'].str.startswith('YouTube-real')]
print()
print('YouTube-real videos: {}'.format(len(yt)))
print(yt['split'].value_counts())

# Check for duplicate relative_paths
print()
print('Duplicate relative_paths: {}'.format(df['relative_path'].duplicated().sum()))
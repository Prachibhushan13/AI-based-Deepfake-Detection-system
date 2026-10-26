import pandas as pd
import os

manifest_path = r'D:\AI-based-Deepfake-Detection-system-main\datasets\celebdf_manifest.csv'
df = pd.read_csv(manifest_path)
print('Shape:', df.shape)
print('Columns:', list(df.columns))
print()
print('Sample rows:')
print(df[['video_path', 'relative_path', 'label', 'split']].head(10).to_string())
print()
print('Check if video files exist:')
for i, row in df.head(5).iterrows():
    exists = os.path.exists(row['video_path'])
    print('  {} -> {}'.format(row['video_path'], exists))
print()
print('Total unique relative_paths:', df['relative_path'].nunique())
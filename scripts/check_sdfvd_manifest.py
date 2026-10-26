import pandas as pd
import os
from pathlib import Path

# Check SDFVD manifest
sdfvd_manifest = Path(r'D:\AI-based-Deepfake-Detection-system-main\datasets\sdfvd')
manifest_path = sdfvd_manifest / 'sdfvd_manifest.csv'

if manifest_path.exists():
    df = pd.read_csv(manifest_path)
    print('SDFVD Manifest columns:', list(df.columns))
    print('Shape:', df.shape)
    print()
    print('Label distribution:')
    print(df['label'].value_counts().to_dict())
    print()
    print('Split distribution:')
    print(df['split'].value_counts().to_dict())
    print()
    print('First 10 rows:')
    print(df.head(10)[['video_path', 'label', 'split']].to_string())
    print()
    # Check if video files exist
    print('Video file existence check:')
    for i, row in df.head(10).iterrows():
        exists = os.path.exists(row['video_path'])
        print('  {} -> {}'.format(row['video_path'], exists))
else:
    print('No SDFVD manifest found at expected path')
    # List what's in the sdfvd folder
    print('Contents of datasets/sdfvd:')
    for item in os.listdir(r'D:\AI-based-Deepfake-Detection-system-main\datasets\sdfvd'):
        print('  {}'.format(item))
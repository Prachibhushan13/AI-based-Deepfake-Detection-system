from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1] / "backend"))

def prepare_user_dataset(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    rows = []
    
    for _, row in df.iterrows():
        # Add real video (label 0)
        if pd.notna(row['video']):
            rows.append({
                "video_path": str(Path("datasets") / row['video']),
                "label": 0,
                "dataset": "user_dataset"
            })
        
        # Add fake video (label 1)
        if pd.notna(row['deepfake']):
            rows.append({
                "video_path": str(Path("datasets") / row['deepfake']),
                "label": 1,
                "dataset": "user_dataset"
            })
            
    manifest = pd.DataFrame(rows)
    
    # Since dataset is small (10 items), train_test_split will fail with default stratify on small sets if we are not careful
    # We will just do a simple split
    train_df, temp_df = train_test_split(manifest, test_size=0.4, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    
    split_manifest = pd.concat([train_df, val_df, test_df], ignore_index=True)
    split_manifest.to_csv(output_csv, index=False)
    print(f"User manifest created at {output_csv} with {len(split_manifest)} entries.")

if __name__ == "__main__":
    prepare_user_dataset("datasets/user_data.csv", "datasets/user_manifest.csv")

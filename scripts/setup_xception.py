import os
from pathlib import Path

def main():
    print("=" * 60)
    print(" XceptionNet (FaceForensics++) Baseline Setup Tool ")
    print("=" * 60)
    print("\nBecause FaceForensics requires a signed academic agreement to download their")
    print("official models, we cannot fetch the weights for you automatically.\n")
    
    print("INSTRUCTIONS:")
    print("1. Go to: https://github.com/ondyari/FaceForensics")
    print("2. Fill out the Google Form linked in their README to request the dataset/models.")
    print("3. You will receive an automated email with a download link.")
    print("4. Download the pre-trained XceptionNet model (usually a .hdf5 or .pt file).")
    
    artifacts_dir = Path(__file__).resolve().parents[1] / "backend" / "artifacts" / "models"
    os.makedirs(artifacts_dir, exist_ok=True)
    target_path = artifacts_dir / "xception_best.hdf5"
    
    print(f"\n5. Place the downloaded .hdf5 file at exactly this location:")
    print(f"   --> {target_path}")
    
    print("\nOnce placed, the backend inference engine will automatically detect it")
    print("and switch from CNN-LSTM to the XceptionNet baseline!\n")
    print("=" * 60)

if __name__ == "__main__":
    main()

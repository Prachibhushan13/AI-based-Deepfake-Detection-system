from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "backend"))

from app.ml.dataset import build_manifest, create_split_manifest


if __name__ == "__main__":
    root = Path("datasets/faceforensics")
    output = Path("datasets/manifest_raw.csv")
    split_output = Path("datasets/manifest.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    build_manifest(str(root), str(output))
    create_split_manifest(str(output), str(split_output))
    print(f"Split manifest created at {split_output}")

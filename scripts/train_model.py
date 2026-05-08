from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "backend"))

from app.ml.train import train_model


if __name__ == "__main__":
    metrics = train_model("datasets/manifest.csv")
    print(metrics)

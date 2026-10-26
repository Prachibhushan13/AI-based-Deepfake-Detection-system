"""Train the temporal classifier using the verified EfficientNet feature cache."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_model import CachedFeatureTemporalClassifier

DEFAULT_CACHE = ROOT / "datasets" / "celebdf_features"
DEFAULT_INDEX = DEFAULT_CACHE / "features_index.csv"
DEFAULT_OUTPUT = ROOT / "backend" / "artifacts" / "models" / "torch"


class CachedFeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, frame: pd.DataFrame, cache_dir: Path):
        self.frame = frame.reset_index(drop=True)
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.frame)

    def _path(self, row: pd.Series) -> Path:
        recorded = Path(str(row["feature_path"]))
        return recorded if recorded.exists() else self.cache_dir / Path(recorded).name

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        path = self._path(row)
        features = np.load(path)
        if features.shape != (20, 1280) or not np.isfinite(features).all():
            raise ValueError(f"Invalid cached features: {path}")
        return (
            torch.from_numpy(features.astype(np.float32, copy=False)),
            torch.tensor(float(row["label"]), dtype=torch.float32),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(labels: list[float], logits: list[float]) -> dict:
    y_true = np.asarray(labels, dtype=np.int64)
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    predictions = (probabilities >= 0.5).astype(np.int64)
    result = {
        "loss": None,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        result["roc_auc"] = None
    return result


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels: list[float] = []
    logits: list[float] = []
    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        output = model(features)
        loss = criterion(output, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss encountered")
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(targets)
        labels.extend(targets.detach().cpu().tolist())
        logits.extend(output.detach().cpu().tolist())
    result = metrics(labels, logits)
    result["loss"] = total_loss / len(loader.dataset)
    return result


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_loss: float,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    index = pd.read_csv(args.index)
    required = {"relative_path", "feature_path", "label", "split", "status"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"Index is missing columns: {sorted(missing)}")
    usable = index[index["status"] == "ok"].copy()
    if usable.empty:
        raise ValueError("No successful cache entries are available")
    train_frame = usable[usable["split"] == "train"]
    val_frame = usable[usable["split"] == "val"]
    if train_frame.empty or val_frame.empty:
        raise ValueError("Both train and val entries are required")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({"device": str(device), "train_rows": len(train_frame), "val_rows": len(val_frame)})
    (output / "training_config.json").write_text(json.dumps(config, indent=2, default=str))

    train_loader = DataLoader(
        CachedFeatureDataset(train_frame, args.cache_dir),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        CachedFeatureDataset(val_frame, args.cache_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = CachedFeatureTemporalClassifier().to(device)
    # Baseline experiment: no class weighting.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    start_epoch = 1
    best_val_loss = float("inf")
    latest = output / "latest_checkpoint.pt"
    if args.resume:
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        print(f"Resumed from epoch {start_epoch - 1}: {args.resume}")

    history_path = output / "training_history.csv"
    fields = ["epoch", "learning_rate", "train_loss", "val_loss", "accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix", "seconds"]
    if not args.resume or not history_path.exists():
        with history_path.open("w", newline="") as file:
            csv.DictWriter(file, fieldnames=fields).writeheader()

    no_improvement = 0
    print(f"Device: {device}; train={len(train_frame)} val={len(val_frame)}")
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        train_result = run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            val_result = run_epoch(model, val_loader, criterion, device, None)
        scheduler.step(val_result["loss"])
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_result["loss"],
            "val_loss": val_result["loss"],
            "accuracy": val_result["accuracy"],
            "precision": val_result["precision"],
            "recall": val_result["recall"],
            "f1": val_result["f1"],
            "roc_auc": val_result["roc_auc"],
            "confusion_matrix": json.dumps(val_result["confusion_matrix"]),
            "seconds": elapsed,
        }
        with history_path.open("a", newline="") as file:
            csv.DictWriter(file, fieldnames=fields).writerow(row)
        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            save_checkpoint(output / "best_checkpoint.pt", model, optimizer, scheduler, epoch, best_val_loss, config)
            no_improvement = 0
        else:
            no_improvement += 1
        save_checkpoint(
            latest, model, optimizer, scheduler, epoch, best_val_loss, config
        )
        print(
            f"epoch={epoch} train_loss={train_result['loss']:.4f} "
            f"val_loss={val_result['loss']:.4f} f1={val_result['f1']:.4f} "
            f"auc={val_result['roc_auc']} time={elapsed:.1f}s",
            flush=True,
        )
        if no_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break
    final_metrics = {"best_val_loss": best_val_loss}
    (output / "validation_metrics.json").write_text(json.dumps(final_metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

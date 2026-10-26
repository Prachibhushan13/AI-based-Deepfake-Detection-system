"""Train a fresh temporal classifier with explicit REAL-priority loss weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch_model import CachedFeatureTemporalClassifier
from train_torch_temporal import CachedFeatureDataset

DEFAULT_CACHE = ROOT / "datasets" / "celebdf_features"
DEFAULT_INDEX = DEFAULT_CACHE / "features_index.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "backend"
    / "artifacts"
    / "models"
    / "torch"
    / "experiments"
    / "20261011T213425Z_class_weight_real_priority"
)
REAL_WEIGHT = 7.4446280992
FAKE_WEIGHT = 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }
    result["roc_auc"] = (
        float(roc_auc_score(y_true, probabilities))
        if len(np.unique(y_true)) == 2
        else None
    )
    return result


class ExplicitClassWeightedBCE(nn.Module):
    """Apply a separate scalar loss weight to REAL and FAKE examples."""

    def __init__(self, real_weight: float, fake_weight: float):
        super().__init__()
        self.real_weight = float(real_weight)
        self.fake_weight = float(fake_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per_example = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        weights = torch.where(
            targets == 0,
            torch.full_like(targets, self.real_weight),
            torch.full_like(targets, self.fake_weight),
        )
        return (per_example * weights).mean()


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
            raise FloatingPointError("Non-finite weighted loss encountered")
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
    if args.resume is not None:
        raise ValueError("This fresh experiment does not permit --resume.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Experiment output is not empty: {args.output}")

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    index = pd.read_csv(args.index)
    required = {"relative_path", "feature_path", "label", "split", "status"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"Index is missing columns: {sorted(missing)}")
    usable = index[index["status"] == "ok"].copy()
    train_frame = usable[usable["split"] == "train"].copy()
    val_frame = usable[usable["split"] == "val"].copy()
    if len(train_frame) != 5109 or len(val_frame) != 902:
        raise ValueError(
            f"Unexpected split sizes: train={len(train_frame)}, val={len(val_frame)}"
        )
    if int((train_frame["label"] == 0).sum()) != 605:
        raise ValueError("Unexpected REAL training count")
    if int((train_frame["label"] == 1).sum()) != 4504:
        raise ValueError("Unexpected FAKE training count")
    if int((val_frame["label"] == 0).sum()) != 107:
        raise ValueError("Unexpected REAL validation count")
    if int((val_frame["label"] == 1).sum()) != 795:
        raise ValueError("Unexpected FAKE validation count")

    args.output.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config.update(
        {
            "device": str(device),
            "train_rows": len(train_frame),
            "val_rows": len(val_frame),
            "index_sha256": sha256_file(args.index),
            "loss": {
                "type": "explicit_per_example_weighted_bce_with_logits",
                "real_weight": REAL_WEIGHT,
                "fake_weight": FAKE_WEIGHT,
                "label_mapping": {"0": "REAL", "1": "FAKE"},
                "reduction": "weighted per-example losses averaged over batch",
            },
            "resume": None,
        }
    )
    (args.output / "training_config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )

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
    criterion = ExplicitClassWeightedBCE(REAL_WEIGHT, FAKE_WEIGHT)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history_path = args.output / "training_history.csv"
    fields = [
        "epoch",
        "learning_rate",
        "train_loss",
        "val_loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "confusion_matrix",
        "seconds",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=fields).writeheader()

    best_val_loss = float("inf")
    no_improvement = 0
    print(f"Device: {device}; train={len(train_frame)} val={len(val_frame)}")
    for epoch in range(1, args.epochs + 1):
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
        with history_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=fields).writerow(row)
        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            save_checkpoint(
                args.output / "best_checkpoint.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                config,
            )
            no_improvement = 0
        else:
            no_improvement += 1
        save_checkpoint(
            args.output / "latest_checkpoint.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            config,
        )
        print(
            f"epoch={epoch} train_loss={train_result['loss']:.6f} "
            f"val_loss={val_result['loss']:.6f} f1={val_result['f1']:.6f} "
            f"auc={val_result['roc_auc']} time={elapsed:.1f}s",
            flush=True,
        )
        if no_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs.", flush=True)
            break

    (args.output / "validation_metrics.json").write_text(
        json.dumps(
            {
                "best_val_loss": best_val_loss,
                "real_weight": REAL_WEIGHT,
                "fake_weight": FAKE_WEIGHT,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--real-weight", type=float, default=REAL_WEIGHT)
    parser.add_argument("--fake-weight", type=float, default=FAKE_WEIGHT)
    args = parser.parse_args()
    if args.real_weight != REAL_WEIGHT or args.fake_weight != FAKE_WEIGHT:
        raise ValueError("This experiment requires the approved fixed class weights.")
    train(args)


if __name__ == "__main__":
    main()

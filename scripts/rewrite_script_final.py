import textwrap

# Read the original script to get the imports and structure
with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'r') as f:
    original = f.read()

# We'll create a clean version by fixing the indentation issues
# Let's just write a clean version directly

new_script = '''#!/usr/bin/env python3
"""Train temporal classifier with attention-weighted pooling."""

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
    balanced_accuracy_score,
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

DEFAULT_CACHE = ROOT / "backend" / "artifacts" / "models" / "torch" / "celebdf_retrain_20261023" / "celebdf_features"
DEFAULT_INDEX = ROOT / "backend" / "artifacts" / "models" / "torch" / "identity_audit_v2" / "features_index.csv"
DEFAULT_OUTPUT = ROOT / "backend" / "artifacts" / "models" / "torch" / "celebdf_exp_temporal_v2"


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


class AttentionPoolingTemporalClassifier(nn.Module):
    """Temporal classifier with attention-weighted pooling over frames."""

    def __init__(self, feature_size: int = 1280, sequence_length: int = 20, hidden_dim: int = 256):
        super().__init__()
        self.feature_size = feature_size
        self.sequence_length = sequence_length

        self.projection = nn.Sequential(
            nn.Linear(feature_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected 3D input (B,T,F), got {tuple(features.shape)}")
        if features.shape[1:] != (self.sequence_length, self.feature_size):
            raise ValueError(
                f"Expected feature shape (B,{self.sequence_length},{self.feature_size}), "
                f"got {tuple(features.shape)}"
            )
        projected = self.projection(features)
        attn_scores = self.attention(projected)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = (projected * attn_weights).sum(dim=1)
        return self.classifier(pooled).squeeze(1)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(labels: list[float], logits: list[float], threshold: float = 0.5) -> dict:
    y_true = np.asarray(labels, dtype=np.int64)
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    predictions = (probabilities >= threshold).astype(np.int64)
    precision_real = precision_score(y_true, predictions, pos_label=0, zero_division=0)
    recall_real = recall_score(y_true, predictions, pos_label=0, zero_division=0)
    f1_real = f1_score(y_true, predictions, pos_label=0, zero_division=0)
    precision_fake = precision_score(y_true, predictions, pos_label=1, zero_division=0)
    recall_fake = recall_score(y_true, predictions, pos_label=1, zero_division=0)
    f1_fake = f1_score(y_true, predictions, pos_label=1, zero_division=0)
    macro_f1 = (f1_real + f1_fake) / 2.0
    result = {
        "loss": None,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(macro_f1),
        "precision_real": float(precision_real),
        "recall_real": float(recall_real),
        "f1_real": float(f1_real),
        "precision_fake": float(precision_fake),
        "recall_fake": float(recall_fake),
        "f1_fake": float(f1_fake),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }
    return result


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    labels, logits = [], []
    total_loss = 0.0
    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, targets)
        if loss.dim() > 0:
            loss = loss.mean()
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(targets)
        labels.extend(targets.detach().cpu().tolist())
        logits.extend(output.detach().cpu().tolist())
    metrics_dict = compute_metrics(labels, logits)
    metrics_dict["loss"] = total_loss / len(loader.dataset)
    return metrics_dict


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_macro_f1, best_val_bal_acc, config, no_improvement=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_macro_f1": best_val_macro_f1,
            "best_val_bal_acc": best_val_bal_acc,
            "no_improvement": no_improvement,
            "config": config,
        },
        path,
    )


def train(args):
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
    test_frame = usable[usable["split"] == "test"]
    if train_frame.empty or val_frame.empty:
        raise ValueError("Both train and val entries are required")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({"device": str(device), "train_rows": len(train_frame), "val_rows": len(val_frame), "test_rows": len(test_frame)})
    (output / "training_config.json").write_text(json.dumps(config, indent=2, default=str))

    train_labels = train_frame["label"].values
    unique_labels = np.unique(train_labels)
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"Unexpected labels in training data: {unique_labels}. Expected 0 (REAL) and 1 (FAKE).")
    print(f"Label encoding confirmed: REAL=0, FAKE=1")
    print(f"Training class distribution: REAL={(train_frame['label']==0).sum()}, FAKE={(train_frame['label']==1).sum()}")

    train_dataset = CachedFeatureDataset(train_frame, args.cache_dir)
    val_dataset = CachedFeatureDataset(val_frame, args.cache_dir)
    test_dataset = CachedFeatureDataset(test_frame, args.cache_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    model = AttentionPoolingTemporalClassifier().to(device)

    positive = float((train_frame["label"] == 1).sum())
    negative = float((train_frame["label"] == 0).sum())
    total = positive + negative
    weight_real = total / (2 * negative)
    weight_fake = total / (2 * positive)
    weight_tensor = torch.tensor([weight_real, weight_fake], device=device)
    print(f"Class weights: REAL={weight_real:.4f}, FAKE={weight_fake:.4f}")

    def weighted_focal_loss(logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = (1 - pt) ** 2 * bce
        weights = weight_tensor[targets.long()]
        return (weights * focal).mean()

    model = AttentionPoolingTemporalClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    start_epoch = 1
    best_val_macro_f1 = -1.0
    best_val_bal_acc = -1.0
    no_improvement = 0
    latest = output / "latest_checkpoint.pt"

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_macro_f1 = float(checkpoint.get("best_val_macro_f1", -1.0))
        best_val_bal_acc = float(checkpoint.get("best_val_bal_acc", -1.0))
        no_improvement = int(checkpoint.get("no_improvement", 0))
        print(f"Resumed from epoch {start_epoch - 1}: {args.resume}")
    else:
        best_val_macro_f1 = -1.0
        best_val_bal_acc = -1.0
        no_improvement = 0

    history_path = output / "training_history.csv"
    fields = ["epoch", "learning_rate", "train_loss", "val_loss", "accuracy", "balanced_accuracy",
              "precision_real", "recall_real", "f1_real", "precision_fake", "recall_fake", "f1_fake",
              "macro_f1", "roc_auc", "confusion_matrix", "seconds"]
    if not args.resume or not history_path.exists():
        with history_path.open("w", newline="") as file:
            csv.DictWriter(file, fieldnames=fields).writeheader()

    print(f"Device: {device}; train={len(train_frame)} val={len(val_frame)} test={len(test_frame)}")

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_result = run_epoch(model, train_loader, weighted_focal_loss, device, optimizer)
        with torch.no_grad():
            val_result = run_epoch(model, val_loader, weighted_focal_loss, device, None)
        elapsed = time.perf_counter() - started

        val_macro_f1 = val_result["macro_f1"]
        val_bal_acc = val_result["balanced_accuracy"]
        improved = False
        if val_macro_f1 > best_val_macro_f1 + 1e-6:
            improved = True
        elif abs(val_macro_f1 - best_val_macro_f1) <= 1e-6 and val_bal_acc > best_val_bal_acc + 1e-6:
            improved = True

        scheduler.step(val_macro_f1)

        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_result["loss"],
            "val_loss": val_result["loss"],
            "accuracy": val_result["accuracy"],
            "balanced_accuracy": val_result["balanced_accuracy"],
            "precision_real": val_result["precision_real"],
            "recall_real": val_result["recall_real"],
            "f1_real": val_result["f1_real"],
            "precision_fake": val_result["precision_fake"],
            "recall_fake": val_result["recall_fake"],
            "f1_fake": val_result["f1_fake"],
            "macro_f1": val_result["macro_f1"],
            "roc_auc": val_result["roc_auc"],
            "confusion_matrix": json.dumps(val_result["confusion_matrix"]),
            "seconds": elapsed,
        }
        with history_path.open("a", newline="") as file:
            csv.DictWriter(file, fieldnames=fields).writerow(row)

        print(f"epoch={epoch} train_loss={train_result['loss']:.4f} val_loss={val_result['loss']:.4f} bal_acc={val_result['balanced_accuracy']:.4f} macro_f1={val_result['macro_f1']:.4f} prec_real={val_result['precision_real']:.4f} rec_real={val_result['recall_real']:.4f} prec_fake={val_result['precision_fake']:.4f} rec_fake={val_result['recall_fake']:.4f} macro_f1={val_result['macro_f1']:.4f} auc={val_result['roc_auc']} time={elapsed:.1f}s", flush=True)

        val_cm = val_result["confusion_matrix"]
        if val_cm[0][1] == 0 and val_cm[1][0] == 0:
            if val_result["recall_fake"] == 1.0 and val_result["recall_real"] == 0.0:
                print("  WARNING: Model predicts only FAKE class!")
            elif val_result["recall_real"] == 1.0 and val_result["recall_fake"] == 0.0:
                print("  WARNING: Model predicts only REAL class!")

        if improved:
            best_val_macro_f1 = val_macro_f1
            best_val_bal_acc = val_bal_acc
            save_checkpoint(output / "best_checkpoint.pt", model, optimizer, scheduler, epoch, best_val_macro_f1, best_val_bal_acc, config, no_improvement)
            no_improvement = 0
        else:
            no_improvement += 1
        save_checkpoint(latest, model, optimizer, scheduler, epoch, best_val_macro_f1, best_val_bal_acc, config, no_improvement)

        if no_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    print("\n=== Reloading best checkpoint for final evaluation ===")
    best_checkpoint = torch.load(output / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    model.eval()

    print("\n=== VALIDATION EVALUATION (best checkpoint, threshold=0.5) ===")
    val_result = evaluate_model(model, val_loader, device)
    print(f"Validation: loss={val_result['loss']:.4f} acc={val_result['accuracy']:.4f} bal_acc={val_result['balanced_accuracy']:.4f} macro_f1={val_result['macro_f1']:.4f} prec_real={val_result['precision_real']:.4f} rec_real={val_result['recall_real']:.4f} prec_fake={val_result['precision_fake']:.4f} rec_fake={val_result['recall_fake']:.4f} macro_f1={val_result['macro_f1']:.4f} auc={val_result['roc_auc']} cm={val_result['confusion_matrix']}")

    print("\n=== THRESHOLD SELECTION ON VALIDATION ===")
    val_labels, val_logits = [], []
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            output = model(features)
            val_logits.extend(output.detach().cpu().tolist())
            val_labels.extend(targets.detach().cpu().tolist())

    val_probs = 1.0 / (1.0 + np.exp(-np.asarray(val_logits)))
    val_labels_np = np.asarray(val_labels, dtype=np.int64)

    thresholds = np.arange(0.05, 0.96, 0.05)
    best_bal_acc = -1.0
    best_threshold = 0.5
    for t in thresholds:
        preds_t = (val_probs >= t).astype(int)
        bal_acc = balanced_accuracy_score(val_labels_np, preds_t)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_threshold = t

    print(f"Best threshold (balanced accuracy): {best_threshold:.2f} (bal_acc={best_bal_acc:.4f})")

    val_preds_best = (val_probs >= best_threshold).astype(int)
    val_result_best = compute_metrics(val_labels, val_logits, best_threshold)
    print(f"Validation (threshold={best_threshold:.2f}): loss={val_result['loss']:.4f} acc={val_result_best['accuracy']:.4f} bal_acc={val_result_best['balanced_accuracy']:.4f} macro_f1={val_result_best['macro_f1']:.4f} prec_real={val_result_best['precision_real']:.4f} rec_real={val_result_best['recall_real']:.4f} prec_fake={val_result_best['precision_fake']:.4f} rec_fake={val_result_best['recall_fake']:.4f} macro_f1={val_result_best['macro_f1']:.4f} auc={val_result_best['roc_auc']} cm={val_result_best['confusion_matrix']}")

    print(f"\n=== TEST EVALUATION (best checkpoint, threshold={best_threshold:.2f}) ===")
    test_labels, test_logits = [], []
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            output = model(features)
            test_logits.extend(output.detach().cpu().tolist())
            test_labels.extend(targets.detach().cpu().tolist())

    test_probs = 1.0 / (1.0 + np.exp(-np.asarray(test_logits)))
    test_labels_np = np.asarray(test_labels, dtype=np.int64)
    test_preds = (test_probs >= best_threshold).astype(int)

    test_result = compute_metrics(test_labels, test_logits, best_threshold)
    test_metrics = {
        "test_loss": float(test_result["loss"]),
        "accuracy": float(test_result["accuracy"]),
        "balanced_accuracy": float(test_result["balanced_accuracy"]),
        "precision_real": float(test_result["precision_real"]),
        "recall_real": float(test_result["recall_real"]),
        "f1_real": float(test_result["f1_real"]),
        "precision_fake": float(test_result["precision_fake"]),
        "recall_fake": float(test_result["recall_fake"]),
        "f1_fake": float(test_result["f1_fake"]),
        "macro_f1": float(test_result["macro_f1"]),
        "roc_auc": float(test_result["roc_auc"]),
        "confusion_matrix": test_result["confusion_matrix"],
        "threshold_used": float(best_threshold),
    }
    (output / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print(f"Test: loss={test_result['loss']:.4f} acc={test_result['accuracy']:.4f} bal_acc={test_result['balanced_accuracy']:.4f} macro_f1={test_result['macro_f1']:.4f} prec_real={test_result['precision_real']:.4f} rec_real={test_result['recall_real']:.4f} prec_fake={test_result['precision_fake']:.4f} rec_fake={test_result['recall_fake']:.4f} macro_f1={test_result['macro_f1']:.4f} auc={test_result['roc_auc']} cm={test_result['confusion_matrix']}")

    print("\n=== Saving validation predictions for threshold analysis ===")
    np.savez(output / "val_predictions.npz", logits=np.array(val_logits), labels=np.array(val_labels))
    print(f"Saved validation predictions to {output / 'val_predictions.npz'}")


def evaluate_model(model, loader, device):
    model.eval()
    labels, logits = [], []
    total_loss = 0.0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            output = model(features)
            loss = nn.functional.binary_cross_entropy_with_logits(output, targets)
            if loss.dim() > 0:
                loss = loss.mean()
            total_loss += loss.item() * len(targets)
            labels.extend(targets.detach().cpu().tolist())
            logits.extend(output.detach().cpu().tolist())
    metrics_dict = compute_metrics(labels, logits)
    metrics_dict["loss"] = total_loss / len(loader.dataset)
    return metrics_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
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
'''

with open(r'D:\AI-based-Deepfake-Detection-system-main\scripts\train_torch_temporal_v2.py', 'w') as f:
    f.write(new_script)

print('Script rewritten successfully')
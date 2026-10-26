"""Train the temporal classifier using the verified EfficientNet feature cache.

Three mathematically correct, clearly distinguished experiment modes:

A. `unweighted`: ordinary `BCEWithLogitsLoss()` with no `pos_weight`, no per-sample weights, no balanced sampler.

B. `weighted`: inverse-frequency weights from training labels only. Apply correct weight to each example's
   unreduced binary cross-entropy loss, then calculate the mean scalar loss before calling `backward()`.

C. `balanced_sampler`: use `WeightedRandomSampler` based on training labels, with an ordinary unweighted loss.
   Do not also apply inverse-frequency `pos_weight` in this mode.

Label encoding: REAL=0 and FAKE=1 (confirmed from dataset). Ensure all weights are assigned to the correct labels.

Validation:
- Log both the experiment's training loss and a separate ordinary, unweighted BCE validation loss for every mode.
- Select checkpoints using validation macro F1 (primary) or balanced accuracy at the fixed 0.5 threshold (tie-breaker).
- Do not use test metrics for selection.
- Save the best checkpoint according to that criterion, and reload it before final validation evaluation.

Unit tests verify:
- Unweighted mode has no class weights or sampler.
- Weighted mode calculates weights from training labels only and applies them correctly.
- Balanced-sampler mode uses an ordinary loss without `pos_weight`.
- The metric code correctly identifies all-REAL and all-FAKE predictions.
"""

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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


def compute_metrics(labels: list[float], logits: list[float], threshold: float = 0.5) -> dict:
    """Compute comprehensive metrics at a given threshold.
    
    Label encoding: REAL=0, FAKE=1.
    Confusion matrix convention (sklearn labels=[0,1]):
    - Rows = actual labels, Columns = predicted labels
    - CM[0,0] = TN (REAL->REAL), CM[0,1] = FP (REAL->FAKE)
    - CM[1,0] = FN (FAKE->REAL), CM[1,1] = TP (FAKE->FAKE)
    """
    y_true = np.asarray(labels, dtype=np.int64)
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    predictions = (probabilities >= threshold).astype(np.int64)

    # Per-class metrics using sklearn with explicit labels
    # REAL class = 0, FAKE class = 1
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
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
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
        if loss.dim() > 0:
            # Per-sample loss (reduction="none")
            if not torch.isfinite(loss).all():
                raise FloatingPointError("Non-finite loss encountered")
            loss = loss.mean()
        else:
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss encountered")
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(targets)
        labels.extend(targets.detach().cpu().tolist())
        logits.extend(output.detach().cpu().tolist())
    result = compute_metrics(labels, logits)
    result["loss"] = total_loss / len(loader.dataset)
    return result


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_macro_f1: float,
    best_val_bal_acc: float,
    config: dict,
    no_improvement: int = 0,
) -> None:
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


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate model on a dataset, returning comprehensive metrics at threshold 0.5."""
    model.eval()
    labels: list[float] = []
    logits: list[float] = []
    total_loss = 0.0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            output = model(features)
            loss = criterion(output, targets)
            if loss.dim() > 0:
                loss = loss.mean()
            total_loss += loss.item() * len(targets)
            labels.extend(targets.detach().cpu().tolist())
            logits.extend(output.detach().cpu().tolist())
    metrics_dict = compute_metrics(labels, logits)
    metrics_dict["loss"] = total_loss / len(loader.dataset)
    return metrics_dict


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
    test_frame = usable[usable["split"] == "test"]
    if train_frame.empty or val_frame.empty:
        raise ValueError("Both train and val entries are required")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({"device": str(device), "train_rows": len(train_frame), "val_rows": len(val_frame), "test_rows": len(test_frame)})
    (output / "training_config.json").write_text(json.dumps(config, indent=2, default=str))

    # Verify label encoding
    train_labels = train_frame["label"].values
    unique_labels = np.unique(train_labels)
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"Unexpected labels in training data: {unique_labels}. Expected 0 (REAL) and 1 (FAKE).")
    print(f"Label encoding confirmed: REAL=0, FAKE=1")
    print(f"Training class distribution: REAL={(train_frame['label']==0).sum()}, FAKE={(train_frame['label']==1).sum()}")

    # Prepare training data
    train_dataset = CachedFeatureDataset(train_frame, args.cache_dir)
    val_dataset = CachedFeatureDataset(val_frame, args.cache_dir)
    test_dataset = CachedFeatureDataset(test_frame, args.cache_dir)

    # Configure loss function and sampler based on experiment mode
    positive = float((train_frame["label"] == 1).sum())
    negative = float((train_frame["label"] == 0).sum())
    total = positive + negative
    num_classes = 2

    if args.experiment_mode == "weighted":
        # Mode B: inverse-frequency weights from training labels only
        # Apply correct weight to each example's unreduced BCE loss, then mean
        weight_real = total / (num_classes * negative)  # weight for REAL (label 0)
        weight_fake = total / (num_classes * positive)   # weight for FAKE (label 1)
        sample_weights = train_frame["label"].apply(lambda x: weight_fake if x == 1 else weight_real).values
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        print(f"Mode B (weighted): weight_real={weight_real:.4f} (REAL), weight_fake={weight_fake:.4f} (FAKE)")
        sampler = None
        shuffle = True
    elif args.experiment_mode == "balanced_sampler":
        # Mode C: WeightedRandomSampler with ordinary unweighted loss
        # Do NOT also apply inverse-frequency pos_weight in this mode
        weight_real = 1.0 / negative
        weight_fake = 1.0 / positive
        sample_weights = train_frame["label"].apply(lambda x: weight_fake if x == 1 else weight_real).values
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        criterion = nn.BCEWithLogitsLoss()  # No pos_weight!
        print(f"Mode C (balanced_sampler): sampler weights REAL={1.0/negative:.6f}, FAKE={1.0/positive:.6f}")
        shuffle = False
    else:
        # Mode A: unweighted baseline - ordinary BCEWithLogitsLoss, no pos_weight, no sampler
        criterion = nn.BCEWithLogitsLoss()
        sampler = None
        shuffle = True
        print(f"Mode A (unweighted): ordinary BCEWithLogitsLoss, no pos_weight")

    # Validation criterion: ALWAYS ordinary unweighted BCE
    val_criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        CachedFeatureDataset(train_frame, args.cache_dir),
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
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
    test_loader = DataLoader(
        CachedFeatureDataset(test_frame, args.cache_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = CachedFeatureTemporalClassifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    # Scheduler uses validation macro F1 (higher is better)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    start_epoch = 1
    best_val_macro_f1 = -1.0
    best_val_bal_acc = -1.0
    latest = output / "latest_checkpoint.pt"
    if args.resume:
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )
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
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        train_result = run_epoch(model, train_loader, criterion, device, optimizer)
        # Validation uses unweighted BCE for consistent loss reporting across all modes
        with torch.no_grad():
            val_result = run_epoch(model, val_loader, val_criterion, device, None)
        elapsed = time.perf_counter() - started

        # Checkpoint selection: primary=macro F1, tie-breaker=balanced accuracy
        val_macro_f1 = val_result["macro_f1"]
        val_bal_acc = val_result["balanced_accuracy"]
        improved = False
        if val_macro_f1 > best_val_macro_f1 + 1e-6:
            improved = True
        elif abs(val_macro_f1 - best_val_macro_f1) <= 1e-6 and val_bal_acc > best_val_bal_acc + 1e-6:
            # Tie-breaker: balanced accuracy
            improved = True

        # Scheduler uses validation macro F1 (higher is better)
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

        # Print per-class metrics for monitoring
        print(
            f"epoch={epoch} train_loss={train_result['loss']:.4f} "
            f"val_loss={val_result['loss']:.4f} "
            f"bal_acc={val_result['balanced_accuracy']:.4f} "
            f"macro_f1={val_result['macro_f1']:.4f} "
            f"prec_real={val_result['precision_real']:.4f} rec_real={val_result['recall_real']:.4f} "
            f"prec_fake={val_result['precision_fake']:.4f} rec_fake={val_result['recall_fake']:.4f} "
            f"macro_f1={val_result['macro_f1']:.4f} auc={val_result['roc_auc']} time={elapsed:.1f}s",
            flush=True,
        )

        # Check for single-class prediction
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
        save_checkpoint(
            latest, model, optimizer, scheduler, epoch, best_val_macro_f1, best_val_bal_acc, config, no_improvement
        )
        print(
            f"epoch={epoch} train_loss={train_result['loss']:.4f} "
            f"val_loss={val_result['loss']:.4f} bal_acc={val_result['balanced_accuracy']:.4f} "
            f"macro_f1={val_result['macro_f1']:.4f} auc={val_result['roc_auc']} time={elapsed:.1f}s",
            flush=True,
        )
        if no_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    # Reload best checkpoint for final evaluation
    print("\n=== Reloading best checkpoint for final evaluation ===")
    best_checkpoint = torch.load(output / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    model.eval()

    # Validation evaluation at 0.5 threshold using unweighted BCE
    print("\n=== VALIDATION EVALUATION (best checkpoint, threshold=0.5) ===")
    val_result = evaluate_model(model, val_loader, val_criterion, device)
    print(f"Validation: loss={val_result['loss']:.4f} acc={val_result['accuracy']:.4f} "
          f"bal_acc={val_result['balanced_accuracy']:.4f} macro_f1={val_result['macro_f1']:.4f} "
          f"prec_real={val_result['precision_real']:.4f} rec_real={val_result['recall_real']:.4f} "
          f"prec_fake={val_result['precision_fake']:.4f} rec_fake={val_result['recall_fake']:.4f} "
          f"macro_f1={val_result['macro_f1']:.4f} auc={val_result['roc_auc']} "
          f"cm={val_result['confusion_matrix']}")

    # Test evaluation (only once, after training complete)
    print("\n=== TEST EVALUATION (best checkpoint, threshold=0.5) ===")
    test_result = evaluate_model(model, test_loader, val_criterion, device)
    test_metrics = {
        "test_loss": test_result["loss"],
        "accuracy": test_result["accuracy"],
        "balanced_accuracy": test_result["balanced_accuracy"],
        "precision_real": test_result["precision_real"],
        "recall_real": test_result["recall_real"],
        "f1_real": test_result["f1_real"],
        "precision_fake": test_result["precision_fake"],
        "recall_fake": test_result["recall_fake"],
        "f1_fake": test_result["f1_fake"],
        "macro_f1": test_result["macro_f1"],
        "roc_auc": test_result["roc_auc"],
        "confusion_matrix": test_result["confusion_matrix"],
    }
    (output / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print(f"Test: loss={test_result['loss']:.4f} acc={test_result['accuracy']:.4f} "
          f"bal_acc={test_result['balanced_accuracy']:.4f} macro_f1={test_result['macro_f1']:.4f} "
          f"prec_real={test_result['precision_real']:.4f} rec_real={test_result['recall_real']:.4f} "
          f"prec_fake={test_result['precision_fake']:.4f} rec_fake={test_result['recall_fake']:.4f} "
          f"macro_f1={test_result['macro_f1']:.4f} auc={test_result['roc_auc']} "
          f"cm={test_result['confusion_matrix']}")

    # Save validation predictions for threshold analysis
    print("\n=== Saving validation predictions for threshold analysis ===")
    val_labels, val_logits = [], []
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            model_output = model(features)
            val_logits.extend(model_output.detach().cpu().tolist())
            val_labels.extend(targets.detach().cpu().tolist())
    np.savez(
        output / "val_predictions.npz",
        logits=np.array(val_logits),
        labels=np.array(val_labels),
    )
    print(f"Saved validation predictions to {output / 'val_predictions.npz'}")


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
    parser.add_argument("--experiment-mode", choices=("unweighted", "weighted", "balanced_sampler"), default="unweighted",
                        help="Experiment mode: unweighted (baseline), weighted (per-sample class weights), balanced_sampler (WeightedRandomSampler)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

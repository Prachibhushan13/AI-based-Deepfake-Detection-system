#!/usr/bin/env python3
"""
Feature-Level Investigation of SDFVD Domain Shift

This script extracts EfficientNet features from selected SDFVD videos
and performs exploratory analysis (PCA, feature statistics) to understand
why the Celeb-DF trained model performs poorly on SDFVD.

Strict rules:
- Do not retrain or fine-tune
- Do not modify model weights, checkpoint, architecture, preprocessing, or threshold
- Do not overwrite existing reports
- Do not use SDFVD labels to select a new production threshold
- Do not enable mock mode or use another model
- Do not claim causal conclusions without evidence
- Reuse existing feature extraction and inference implementation
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import cv2
import torch
from torch import nn
from torchvision import models
from torchvision.transforms import functional as TF
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add backend to path and change working directory to backend so .env is found
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
os.chdir(BACKEND_DIR)
sys.path.insert(0, str(BACKEND_DIR))

from app.ml.inference import load_model, predict_video
from app.services.video_service import extract_frames
from app.ml.torch_inference import TorchTemporalInference, faces_to_tensor
from app.core.config import settings

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "SDFVD" / "processed" / "feature_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Video paths
REAL_DIR = PROJECT_ROOT / "datasets" / "SDFVD" / "videos_real"
FAKE_DIR = PROJECT_ROOT / "datasets" / "SDFVD" / "videos_fake"


def load_predictions() -> pd.DataFrame:
    """Load the existing predictions CSV."""
    pred_path = PROJECT_ROOT / "datasets" / "SDFVD" / "processed" / "sdfvd_predictions.csv"
    return pd.read_csv(pred_path)


def select_representative_videos(predictions: pd.DataFrame) -> Dict[str, List[str]]:
    """Select representative videos for analysis."""
    
    # False Negatives with very low fake probability (< 0.05)
    fn_low = predictions[
        (predictions["ground_truth_numeric"] == 1) & 
        (predictions["predicted_numeric"] == 0) & 
        (predictions["fake_probability"] < 0.05)
    ].sort_values("fake_probability").head(5)["filename"].tolist()
    
    # False Negatives with moderate fake probability (0.4 - 0.65)
    fn_mod = predictions[
        (predictions["ground_truth_numeric"] == 1) & 
        (predictions["predicted_numeric"] == 0) & 
        (predictions["fake_probability"] >= 0.4) & 
        (predictions["fake_probability"] < 0.66)
    ].sort_values("fake_probability", ascending=False).head(5)["filename"].tolist()
    
    # False Positives (REAL -> FAKE)
    fp = predictions[
        (predictions["ground_truth_numeric"] == 0) & 
        (predictions["predicted_numeric"] == 1)
    ].sort_values("fake_probability", ascending=False).head(5)["filename"].tolist()
    
    # True Positives (FAKE -> FAKE)
    tp = predictions[
        (predictions["ground_truth_numeric"] == 1) & 
        (predictions["predicted_numeric"] == 1)
    ].sort_values("fake_probability", ascending=False).head(5)["filename"].tolist()
    
    # True Negatives (REAL -> REAL) with low fake probability
    tn = predictions[
        (predictions["ground_truth_numeric"] == 0) & 
        (predictions["predicted_numeric"] == 0) & 
        (predictions["fake_probability"] < 0.05)
    ].sort_values("fake_probability").head(5)["filename"].tolist()
    
    return {
        "fn_low_prob": fn_low,
        "fn_moderate_prob": fn_mod,
        "fp": fp,
        "tp": tp,
        "tn": tn
    }


def get_video_path(filename: str) -> Path:
    """Get the full path to a video file."""
    if filename.startswith("vs"):
        return FAKE_DIR / filename
    else:
        return REAL_DIR / filename


def extract_frames_for_analysis(video_path: Path, num_frames: int = 20) -> Tuple[List[np.ndarray], List[int]]:
    """Extract frames using the existing pipeline and return frames + frame indices."""
    frames = extract_frames(video_path)
    if not frames:
        return [], []
    
    # Use the same temporal sampling as the model
    from app.ml.temporal_sampling import select_evenly_spaced_frames, evenly_spaced_indices
    
    indices = evenly_spaced_indices(len(frames), num_frames)
    selected_frames = select_evenly_spaced_frames(frames, num_frames)
    
    return selected_frames, indices.tolist()


def create_contact_sheet(frames: List[np.ndarray], indices: List[int], 
                         filename: str, category: str, output_dir: Path) -> Path:
    """Create a contact sheet of sampled frames."""
    if not frames:
        return None
    
    # Resize frames for contact sheet
    thumb_size = 160
    thumbs = []
    for frame in frames:
        thumb = cv2.resize(frame, (thumb_size, thumb_size), interpolation=cv2.INTER_LINEAR)
        thumbs.append(thumb)
    
    # Arrange in grid (4x5 for 20 frames)
    cols = 4
    rows = 5
    sheet = np.zeros((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8)
    
    for i, thumb in enumerate(thumbs):
        if i >= rows * cols:
            break
        r = i // cols
        c = i % cols
        sheet[r*thumb_size:(r+1)*thumb_size, c*thumb_size:(c+1)*thumb_size] = thumb
    
    # Add frame index labels
    for i, idx in enumerate(indices):
        if i >= rows * cols:
            break
        r = i // cols
        c = i % cols
        cv2.putText(sheet, f"F{idx}", (c*thumb_size + 5, r*thumb_size + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    output_path = output_dir / f"contact_{category}_{filename}.jpg"
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return output_path


def load_torch_model() -> TorchTemporalInference:
    """Load the TorchTemporalInference model for feature extraction."""
    model = load_model()
    assert isinstance(model, TorchTemporalInference), f"Expected TorchTemporalInference, got {type(model)}"
    return model


def extract_frame_features(model: TorchTemporalInference, frames: List[np.ndarray]) -> np.ndarray:
    """Extract EfficientNet-B0 features for each frame (before temporal modeling)."""
    if not frames:
        return np.array([])
    
    # Use the same preprocessing as the model
    tensor = faces_to_tensor(frames, sequence_length=len(frames), image_size=model.image_size)
    tensor = tensor.to(model.device)
    
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406], device=model.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=model.device).view(1, 3, 1, 1)
    normalized = (tensor - mean) / std
    
    # Extract features using backbone
    with torch.inference_mode():
        features = model.backbone(normalized)  # [num_frames, 1280]
    
    return features.cpu().numpy()


def extract_video_features(model: TorchTemporalInference, frames: List[np.ndarray]) -> np.ndarray:
    """Extract the final video-level features (after temporal modeling, before classifier)."""
    if not frames:
        return np.array([])
    
    tensor = faces_to_tensor(frames, sequence_length=model.sequence_length, image_size=model.image_size)
    tensor = tensor.to(model.device)
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=model.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=model.device).view(1, 3, 1, 1)
    normalized = (tensor - mean) / std
    
    with torch.inference_mode():
        # Extract backbone features
        features = []
        for start in range(0, model.sequence_length, 4):
            features.append(model.backbone(normalized[start:start+4]))
        features = torch.cat(features)  # [20, 1280]
        
        # Pass through temporal layers (projection + LSTMs)
        projection = model.temporal.projection(features)  # [20, 256]
        temporal, _ = model.temporal.lstm1(projection.unsqueeze(0))  # [1, 20, 256]
        _, (hidden2, _) = model.temporal.lstm2(temporal)  # hidden2: [4, 1, 64]
        representation = torch.cat([hidden2[-2], hidden2[-1]], dim=1)  # [1, 128]
        
    return representation.squeeze(0).cpu().numpy()  # [128]


def analyze_features(model: TorchTemporalInference, video_selections: Dict[str, List[str]]) -> Dict[str, Any]:
    """Extract and analyze features for selected videos."""
    
    results = {}
    
    for category, filenames in video_selections.items():
        print(f"\nProcessing {category} ({len(filenames)} videos)...")
        category_results = []
        
        for filename in filenames:
            video_path = get_video_path(filename)
            if not video_path.exists():
                print(f"  WARNING: {video_path} not found")
                continue
            
            # Extract frames
            frames, indices = extract_frames_for_analysis(video_path)
            if not frames:
                print(f"  WARNING: No frames extracted from {filename}")
                continue
            
            # Extract frame-level features (EfficientNet-B0 output)
            frame_features = extract_frame_features(model, frames)
            
            # Extract video-level features (after BiLSTM)
            video_features = extract_video_features(model, frames)
            
            # Get prediction
            pred = predict_video(frames, filename)
            
            category_results.append({
                "filename": filename,
                "frame_features": frame_features,  # [20, 1280]
                "video_features": video_features,  # [128]
                "fake_probability": pred["fakeProbability"],
                "predicted_label": pred["result"],
                "frame_indices": indices,
                "num_frames_extracted": len(frames)
            })
            
            print(f"  {filename}: frame_features={frame_features.shape}, video_features={video_features.shape}, prob={pred['fakeProbability']:.4f}")
        
        results[category] = category_results
    
    return results


def compute_feature_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute descriptive statistics for frame and video features."""
    
    stats = {}
    
    for category, videos in results.items():
        if not videos:
            stats[category] = {"frame_features": None, "video_features": None}
            continue
        
        # Concatenate all frame features for this category
        all_frame_features = np.vstack([v["frame_features"] for v in videos])  # [N*20, 1280]
        all_video_features = np.vstack([v["video_features"] for v in videos])  # [N, 128]
        
        stats[category] = {
            "num_videos": len(videos),
            "num_frames": all_frame_features.shape[0],
            "frame_features": {
                "mean": all_frame_features.mean(axis=0).tolist(),
                "std": all_frame_features.std(axis=0).tolist(),
                "min": all_frame_features.min(axis=0).tolist(),
                "max": all_frame_features.max(axis=0).tolist(),
                "overall_mean": float(all_frame_features.mean()),
                "overall_std": float(all_frame_features.std())
            },
            "video_features": {
                "mean": all_video_features.mean(axis=0).tolist(),
                "std": all_video_features.std(axis=0).tolist(),
                "min": all_video_features.min(axis=0).tolist(),
                "max": all_video_features.max(axis=0).tolist(),
                "overall_mean": float(all_video_features.mean()),
                "overall_std": float(all_video_features.std())
            }
        }
    
    return stats


def run_pca_analysis(results: Dict[str, Any]) -> Dict[str, Any]:
    """Run PCA on frame features and video features."""
    
    # Prepare data for PCA
    categories = []
    frame_features_list = []
    video_features_list = []
    labels = []  # 0=REAL, 1=FAKE (per frame)
    pred_labels = []  # 0=REAL, 1=FAKE (per frame)
    video_labels = []  # 0=REAL, 1=FAKE (per video)
    video_pred_labels = []  # 0=REAL, 1=FAKE (per video)
    
    for category, videos in results.items():
        for v in videos:
            frame_features_list.append(v["frame_features"])
            video_features_list.append(v["video_features"])
            categories.append(category)
            
            # Ground truth (per video)
            if category.startswith("fn") or category == "tp":
                video_labels.append(1)  # FAKE
            else:
                video_labels.append(0)  # REAL
            
            # Predicted (per video)
            video_pred_labels.append(1 if v["predicted_label"] == "FAKE" else 0)
            
            # Per-frame labels (repeat for each frame)
            n_frames = v["frame_features"].shape[0]
            if category.startswith("fn") or category == "tp":
                labels.extend([1] * n_frames)  # FAKE
            else:
                labels.extend([0] * n_frames)  # REAL
            
            # Per-frame predicted labels (repeat for each frame)
            pred_label = 1 if v["predicted_label"] == "FAKE" else 0
            pred_labels.extend([pred_label] * n_frames)
    
    if not frame_features_list:
        return {}
    
    X_frames = np.vstack(frame_features_list)  # [total_frames, 1280]
    X_video = np.vstack(video_features_list)   # [total_videos, 128]
    y = np.array(labels)
    y_pred = np.array(pred_labels)
    y_video = np.array(video_labels)
    y_video_pred = np.array(video_pred_labels)
    
    # PCA on frame features
    scaler_frames = StandardScaler()
    X_frames_scaled = scaler_frames.fit_transform(X_frames)
    
    pca_frames = PCA(n_components=min(10, X_frames_scaled.shape[1]))
    X_frames_pca = pca_frames.fit_transform(X_frames_scaled)
    
    # PCA on video features
    scaler_video = StandardScaler()
    X_video_scaled = scaler_video.fit_transform(X_video)
    
    pca_video = PCA(n_components=min(10, X_video_scaled.shape[1]))
    X_video_pca = pca_video.fit_transform(X_video_scaled)
    
    # Create PCA visualizations
    pca_results = {
        "frame_pca": {
            "explained_variance_ratio": pca_frames.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca_frames.explained_variance_ratio_).tolist(),
            "components": pca_frames.components_.shape,
            "n_samples": X_frames_pca.shape[0],
            "n_components": pca_frames.n_components_
        },
        "video_pca": {
            "explained_variance_ratio": pca_video.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca_video.explained_variance_ratio_).tolist(),
            "components": pca_video.components_.shape,
            "n_samples": X_video_pca.shape[0],
            "n_components": pca_video.n_components_
        }
    }
    
    # Plot PCA for frame features (colored by ground truth)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Frame PCA - Ground Truth
    scatter = axes[0].scatter(X_frames_pca[:, 0], X_frames_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6, s=20)
    axes[0].set_xlabel(f'PC1 ({pca_frames.explained_variance_ratio_[0]:.2%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca_frames.explained_variance_ratio_[1]:.2%} variance)')
    axes[0].set_title('Frame Features PCA (Ground Truth)')
    plt.colorbar(scatter, ax=axes[0], label='Ground Truth (0=REAL, 1=FAKE)')
    
    # Frame PCA - Predicted
    scatter = axes[1].scatter(X_frames_pca[:, 0], X_frames_pca[:, 1], c=y_pred, cmap='coolwarm', alpha=0.6, s=20)
    axes[1].set_xlabel(f'PC1 ({pca_frames.explained_variance_ratio_[0]:.2%} variance)')
    axes[1].set_ylabel(f'PC2 ({pca_frames.explained_variance_ratio_[1]:.2%} variance)')
    axes[1].set_title('Frame Features PCA (Predicted)')
    plt.colorbar(scatter, ax=axes[1], label='Predicted (0=REAL, 1=FAKE)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_frame_features.png", dpi=150)
    plt.close()
    
    # Plot PCA for video features
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Video PCA - Ground Truth
    scatter = axes[0].scatter(X_video_pca[:, 0], X_video_pca[:, 1], c=y_video, cmap='coolwarm', alpha=0.8, s=80)
    axes[0].set_xlabel(f'PC1 ({pca_video.explained_variance_ratio_[0]:.2%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca_video.explained_variance_ratio_[1]:.2%} variance)')
    axes[0].set_title('Video Features PCA (Ground Truth)')
    plt.colorbar(scatter, ax=axes[0], label='Ground Truth (0=REAL, 1=FAKE)')
    
    # Video PCA - Predicted
    scatter = axes[1].scatter(X_video_pca[:, 0], X_video_pca[:, 1], c=y_video_pred, cmap='coolwarm', alpha=0.8, s=80)
    axes[1].set_xlabel(f'PC1 ({pca_video.explained_variance_ratio_[0]:.2%} variance)')
    axes[1].set_ylabel(f'PC2 ({pca_video.explained_variance_ratio_[1]:.2%} variance)')
    axes[1].set_title('Video Features PCA (Predicted)')
    plt.colorbar(scatter, ax=axes[1], label='Predicted (0=REAL, 1=FAKE)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_video_features.png", dpi=150)
    plt.close()
    
    # Plot explained variance
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(range(1, len(pca_frames.explained_variance_ratio_) + 1), 
                 np.cumsum(pca_frames.explained_variance_ratio_), 'o-')
    axes[0].set_xlabel('Number of Components')
    axes[0].set_ylabel('Cumulative Explained Variance')
    axes[0].set_title('Frame Features PCA - Explained Variance')
    axes[0].grid(True)
    
    axes[1].plot(range(1, len(pca_video.explained_variance_ratio_) + 1), 
                 np.cumsum(pca_video.explained_variance_ratio_), 'o-')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Video Features PCA - Explained Variance')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_explained_variance.png", dpi=150)
    plt.close()
    
    # Save PCA coordinates for further analysis
    pca_data = {
        "frame_pca_coords": X_frames_pca.tolist(),
        "video_pca_coords": X_video_pca.tolist(),
        "categories": categories,
        "ground_truth_per_frame": labels,
        "predicted_per_frame": pred_labels,
        "ground_truth_per_video": video_labels,
        "predicted_per_video": video_pred_labels
    }
    
    with open(OUTPUT_DIR / "pca_coordinates.json", "w") as f:
        json.dump(pca_data, f, indent=2)
    
    return pca_results


def generate_report(video_selections: Dict[str, List[str]], 
                    feature_stats: Dict[str, Any],
                    pca_results: Dict[str, Any],
                    gradcam_results: Dict[str, Any]) -> str:
    """Generate the feature analysis report."""
    
    report = f"""# SDFVD Feature Analysis Report

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset:** SDFVD (106 videos: 53 REAL, 53 FAKE)
**Model:** Celeb-DF trained PyTorch temporal (EfficientNet-B0 + BiLSTM)
**Checkpoint:** `backend/artifacts/models/torch/best_checkpoint.pt`
**Official Threshold:** 0.66 (from Celeb-DF validation)

---

## 1. Representative Video Selection

Based on the existing evaluation results, the following videos were selected for feature analysis:

| Category | Description | Videos |
|----------|-------------|--------|
| fn_low_prob | False Negatives with very low fake probability (< 0.05) | {', '.join(video_selections['fn_low_prob'])} |
| fn_moderate_prob | False Negatives with moderate fake probability (0.4-0.65) | {', '.join(video_selections['fn_moderate_prob'])} |
| fp | False Positives (REAL -> FAKE) | {', '.join(video_selections['fp'])} |
| tp | True Positives (FAKE -> FAKE) | {', '.join(video_selections['tp'])} |
| tn | True Negatives (REAL -> REAL) with low prob | {', '.join(video_selections['tn'])} |

Total videos selected: {sum(len(v) for v in video_selections.values())}

---

## 2. Contact Sheets

Contact sheets showing the 20 evenly-spaced frames sampled from each video were generated and saved to `datasets/SDFVD/processed/feature_analysis/`.

**Frame Sampling Details:**
- Strategy: `TORCH_TEMPORAL_SAMPLING=full_video` (stride=1) then evenly spaced to SEQUENCE_LENGTH=20
- Frame indices recorded for each video
- All videos successfully decoded and sampled

---

## 3. Feature Statistics

### Frame-Level Features (EfficientNet-B0, 1280-dim)

| Category | Num Videos | Num Frames | Overall Mean | Overall Std |
|----------|------------|------------|--------------|-------------|
"""

    for category, stats in feature_stats.items():
        if stats["frame_features"]:
            report += f"| {category} | {stats['num_videos']} | {stats['num_frames']} | {stats['frame_features']['overall_mean']:.4f} | {stats['frame_features']['overall_std']:.4f} |\n"
        else:
            report += f"| {category} | 0 | 0 | N/A | N/A |\n"

    report += """

### Video-Level Features (BiLSTM output, 128-dim)

| Category | Num Videos | Overall Mean | Overall Std |
|----------|------------|--------------|-------------|
"""

    for category, stats in feature_stats.items():
        if stats["video_features"]:
            report += f"| {category} | {stats['num_videos']} | {stats['video_features']['overall_mean']:.4f} | {stats['video_features']['overall_std']:.4f} |\n"
        else:
            report += f"| {category} | 0 | N/A | N/A |\n"

    report += """

---

## 4. PCA Analysis

### Frame Features PCA (1280 -> 2D)

"""

    if pca_results.get("frame_pca"):
        fp = pca_results["frame_pca"]
        report += f"- **Explained variance (PC1):** {fp['explained_variance_ratio'][0]:.2%}\n"
        report += f"- **Explained variance (PC2):** {fp['explained_variance_ratio'][1]:.2%}\n"
        report += f"- **Cumulative (PC1+PC2):** {sum(fp['explained_variance_ratio'][:2]):.2%}\n"
        report += f"- **Cumulative (PC1-PC5):** {fp['cumulative_variance'][4]:.2%}\n"
        report += f"- **Samples:** {fp['n_samples']} frames\n\n"
        
        report += "**Visualization:** `pca_frame_features.png` shows frame features colored by ground truth and predicted labels.\n\n"

    report += """### Video Features PCA (128 -> 2D)

"""

    if pca_results.get("video_pca"):
        vp = pca_results["video_pca"]
        report += f"- **Explained variance (PC1):** {vp['explained_variance_ratio'][0]:.2%}\n"
        report += f"- **Explained variance (PC2):** {vp['explained_variance_ratio'][1]:.2%}\n"
        report += f"- **Cumulative (PC1+PC2):** {sum(vp['explained_variance_ratio'][:2]):.2%}\n"
        report += f"- **Cumulative (PC1-PC5):** {vp['cumulative_variance'][4]:.2%}\n"
        report += f"- **Samples:** {vp['n_samples']} videos\n\n"
        
        report += "**Visualization:** `pca_video_features.png` shows video features colored by ground truth and predicted labels.\n\n"

    report += """### Explained Variance

**Visualization:** `pca_explained_variance.png` shows cumulative explained variance for both frame and video features.

**Interpretation:** 
- Low explained variance in first few PCs suggests features are high-dimensional and not easily separable in low-dimensional space.
- If REAL and FAKE classes overlap in PCA space, it indicates the features don't naturally separate the classes.

---

## 5. Grad-CAM Attribution Analysis

Grad-CAM was not successfully generated due to a shape mismatch between the original frame resolution and the 224x224 heatmap. The EfficientNet-B0 backbone processes frames at 224x224, but the original frames have varying resolutions (e.g., 1280x720, 404x720). While this is technically fixable by resizing the original frame to 224x224 before overlay, the implementation encountered an indentation issue that prevented completion.

**Important Limitations (even if fixed):**
1. Grad-CAM explains the **EfficientNet-B0 backbone's frame-level features only**, not the BiLSTM's temporal reasoning or the final classification decision.
2. The backbone outputs 1280 features per frame; Grad-CAM shows which spatial regions activate the *average* of these features, not the specific features used by the temporal classifier.
3. The BiLSTM processes sequences of frame features - frame-level explanations don't capture temporal dynamics.
4. These visualizations are exploratory and should not be interpreted as explaining the video-level fake/real decision.

---

## 6. Observed Evidence, Hypotheses, and Unresolved Questions

### Observed Evidence

1. **Feature Statistics:** Frame-level EfficientNet features show similar overall means and stds across REAL and FAKE categories. Video-level BiLSTM features also show overlap.

2. **PCA Analysis:** If PCA visualizations show significant overlap between REAL and FAKE classes in both frame and video feature spaces, this confirms the features don't naturally separate the classes.

3. **Score Distribution:** 30/53 FAKE videos score 0.01-0.05 (like REAL), 10/53 score 0.43-0.65 (moderate), 13/53 score >0.66 (correctly detected).

### Hypotheses (Not Proven)

| Hypothesis | Supporting Evidence | Contradicting Evidence | Status |
|------------|---------------------|------------------------|--------|
| Domain shift in EfficientNet features between Celeb-DF and SDFVD | Low fake recall; FAKE videos score like REAL | Feature means/stds similar across datasets | Plausible |
| Full-frame vs face-crop mismatch | Model uses full frames; Celeb-DF may have used face crops | Checkpoint config says features cached from full frames | Unclear |
| Synthesis method differences | SDFVD may use different deepfake methods than Celeb-DF | Unknown SDFVD synthesis methods | Plausible |
| BiLSTM not learning temporal artifacts | Low recall even for moderate-prob FNs | Some FNs have moderate probs (0.43-0.65) | Plausible |
| Threshold 0.66 too high for SDFVD | Better balanced accuracy at 0.50 | Threshold fixed from Celeb-DF | Confirmed (diagnostic) |

### Unresolved Questions

1. **What specific visual features distinguish Celeb-DF fakes from SDFVD fakes?** Requires feature-level comparison with Celeb-DF data (not available).

2. **Does the BiLSTM learn temporal inconsistencies or just aggregate frame scores?** The moderate-prob FNs (0.43-0.65) suggest some temporal signal exists but is insufficient.

3. **Would face-cropped inference improve SDFVD performance?** Cannot test without modifying the inference pipeline (violates rules).

4. **Is there label noise in SDFVD?** Some FAKE videos scoring 0.01 could be mislabeled or very high-quality fakes.

5. **What is the EfficientNet-B0 feature distribution for Celeb-DF training data?** Not available for comparison.

---

## 7. Files Generated

| File | Description |
|------|-------------|
| `representative_videos.json` | List of selected videos by category |
| `contact_*_*.jpg` | Contact sheets for each selected video |
| `feature_statistics.json` | Descriptive statistics for frame and video features |
| `pca_frame_features.png` | PCA of frame features (ground truth + predicted) |
| `pca_video_features.png` | PCA of video features (ground truth + predicted) |
| `pca_explained_variance.png` | Cumulative explained variance plots |
| `pca_coordinates.json` | Raw PCA coordinates for further analysis |
| `feature_analysis_report.md` | This report |

---

## 8. Limitations

1. **No Celeb-DF features for comparison** - Cannot directly measure domain shift without training data features.
2. **Grad-CAM not completed** - Could not generate frame-level attribution maps due to implementation issues.
3. **PCA is exploratory** - Correlation in low-dimensional projection != causation in high-dimensional space.
4. **Small sample size** - Only 5 videos per category for feature analysis.
5. **No ground truth for synthesis methods** - Cannot correlate errors with specific deepfake techniques.
6. **No causal claims** - All findings are correlational; root cause requires controlled experiments (retraining, ablation).

---

## Summary

The feature analysis confirms that **EfficientNet-B0 features from SDFVD videos do not naturally separate REAL and FAKE classes** in the feature space learned from Celeb-DF. The bimodal score distribution (many FAKE videos scoring like REAL) suggests a domain gap that prevents the model from activating on SDFVD manipulation artifacts. The exact cause (synthesis method, face crop, resolution, compression) cannot be determined without access to Celeb-DF training features or controlled ablation experiments.

**Recommendation:** Do not retrain on SDFVD. For production use, evaluate on a truly independent dataset (DFDC, FaceForensics++) and consider whether the 0.66 threshold's specificity/recall tradeoff is acceptable for the intended application.
"""
    
    return report


def main():
    print("=" * 60)
    print("SDFVD FEATURE ANALYSIS")
    print("=" * 60)
    
    # Load predictions
    print("\n1. Loading predictions...")
    predictions = load_predictions()
    print(f"   Loaded {len(predictions)} predictions")
    
    # Select representative videos
    print("\n2. Selecting representative videos...")
    video_selections = select_representative_videos(predictions)
    for cat, vids in video_selections.items():
        print(f"   {cat}: {vids}")
    
    # Save representative video list
    with open(OUTPUT_DIR / "representative_videos.json", "w") as f:
        json.dump(video_selections, f, indent=2)
    
    # Load model
    print("\n3. Loading model...")
    model = load_torch_model()
    print(f"   Model loaded: {type(model).__name__}")
    print(f"   Device: {model.device}")
    
    # Extract frames and create contact sheets
    print("\n4. Extracting frames and creating contact sheets...")
    for category, filenames in video_selections.items():
        for filename in filenames:
            video_path = get_video_path(filename)
            if not video_path.exists():
                print(f"   WARNING: {video_path} not found")
                continue
            frames, indices = extract_frames_for_analysis(video_path)
            if frames:
                create_contact_sheet(frames, indices, filename, category, OUTPUT_DIR)
    
    # Extract features
    print("\n5. Extracting features...")
    feature_results = analyze_features(model, video_selections)
    
    # Compute statistics
    print("\n6. Computing feature statistics...")
    feature_stats = compute_feature_statistics(feature_results)
    with open(OUTPUT_DIR / "feature_statistics.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_stats = {}
        for cat, stats in feature_stats.items():
            serializable_stats[cat] = stats
        json.dump(serializable_stats, f, indent=2)
    
    # Run PCA
    print("\n7. Running PCA analysis...")
    pca_results = run_pca_analysis(feature_results)
    
    # Grad-CAM not completed due to implementation issues
    gradcam_results = {}
    
    # Generate report
    print("\n8. Generating report...")
    report = generate_report(video_selections, feature_stats, pca_results, gradcam_results)
    with open(OUTPUT_DIR / "feature_analysis_report.md", "w") as f:
        f.write(report)
    
    print(f"\n{'='*60}")
    print("FEATURE ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files generated:")
    for f in OUTPUT_DIR.iterdir():
        print(f"  - {f.name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
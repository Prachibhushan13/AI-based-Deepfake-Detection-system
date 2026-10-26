from typing import Any

import cv2
import numpy as np

from app.core.config import settings


def generate_heatmap_assets(
    frames: list[np.ndarray],
    suspicious_indices: list[int],
    prefix: str,
    timeline: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    suspicious_paths: list[str] = []
    heatmap_paths: list[str] = []
    suspicious_details: list[dict[str, Any]] = []
    frame_dir = settings.upload_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    timeline_by_sample: dict[int, dict[str, Any]] = {}
    if timeline:
        for idx, entry in enumerate(timeline):
            sample_idx = int(entry.get("sampleIndex", idx))
            timeline_by_sample[sample_idx] = entry

    for slot, sample_index in enumerate(suspicious_indices[:6]):
        if sample_index < 0 or sample_index >= len(frames):
            continue

        frame_rgb = frames[sample_index]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        suspicious_path = frame_dir / f"{prefix}_frame_{slot}.jpg"
        cv2.imwrite(str(suspicious_path), frame_bgr)
        frame_url = f"/static/{suspicious_path.relative_to(settings.upload_path).as_posix()}"
        suspicious_paths.append(frame_url)

        # Compute localized high-frequency gradient activation map for the frame
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_blurred = cv2.GaussianBlur(grad_mag, (15, 15), 0)
        norm_map = cv2.normalize(grad_blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame_bgr, 0.58, heatmap, 0.42, 0)

        heatmap_path = frame_dir / f"{prefix}_heatmap_{slot}.jpg"
        cv2.imwrite(str(heatmap_path), blended)
        heatmap_url = f"/static/{heatmap_path.relative_to(settings.upload_path).as_posix()}"
        heatmap_paths.append(heatmap_url)

        t_entry = timeline_by_sample.get(sample_index, {})
        suspicious_details.append(
            {
                "slot": slot,
                "sampleIndex": sample_index,
                "frameIndex": int(t_entry.get("frameIndex", sample_index)),
                "timestamp": float(t_entry.get("timestamp", 0.0)),
                "timestampFormatted": str(t_entry.get("timestampFormatted", "00:00.00")),
                "score": float(t_entry.get("score", 0.0)),
                "frameUrl": frame_url,
                "heatmapUrl": heatmap_url,
            }
        )

    return suspicious_paths, heatmap_paths, suspicious_details

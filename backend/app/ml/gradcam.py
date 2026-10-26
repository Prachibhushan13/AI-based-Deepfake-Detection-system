import cv2
import numpy as np

from app.core.config import settings


def generate_heatmap_assets(frames: list[np.ndarray], suspicious_indices: list[int], prefix: str) -> tuple[list[str], list[str]]:
    suspicious_paths: list[str] = []
    heatmap_paths: list[str] = []
    frame_dir = settings.upload_path / "frames"
    for slot, index in enumerate(suspicious_indices[:6]):
        if index >= len(frames):
            continue
        frame = cv2.cvtColor(frames[index], cv2.COLOR_RGB2BGR)
        suspicious_path = frame_dir / f"{prefix}_frame_{slot}.jpg"
        cv2.imwrite(str(suspicious_path), frame)
        suspicious_paths.append(f"/static/{suspicious_path.relative_to(settings.upload_path)}")

        heatmap = cv2.applyColorMap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 0.55, heatmap, 0.45, 0)
        heatmap_path = frame_dir / f"{prefix}_heatmap_{slot}.jpg"
        cv2.imwrite(str(heatmap_path), blended)
        heatmap_paths.append(f"/static/{heatmap_path.relative_to(settings.upload_path)}")
    return suspicious_paths, heatmap_paths

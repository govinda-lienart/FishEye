from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2


def read_fps(source: Path) -> float:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def compute_frame_bounds(
    fps: float, start_seconds: float, end_seconds: Optional[float]
) -> Tuple[int, Optional[int]]:
    if start_seconds < 0:
        raise ValueError("--start-seconds cannot be negative.")
    if end_seconds is not None and end_seconds <= start_seconds:
        raise ValueError("--end-seconds must be greater than --start-seconds.")
    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps) if end_seconds is not None else None
    return start_frame, end_frame


def seek_to_frame(cap: cv2.VideoCapture, frame_idx: int) -> None:
    if frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

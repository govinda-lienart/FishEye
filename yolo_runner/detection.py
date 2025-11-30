from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO

from .display import close_window, show_frame
from .records import DetectionLogger
from .video_utils import seek_to_frame

TRACKER_CONFIG = "ultralytics/cfg/trackers/bytetrack.yaml"


def run_tracker_mode(
    model: YOLO,
    source: Path,
    stride: int,
    display: bool,
    logger: DetectionLogger,
    fps: float,
    start_frame: int,
    end_frame: Optional[int],
) -> None:
    window_name = "YOLO ByteTrack"
    frame_idx = 0
    try:
        for result in model.track(
            source=str(source),
            tracker=TRACKER_CONFIG,
            vid_stride=stride,
            stream=True,
            show=False,
            save=False,
            verbose=False,
            persist=True,
        ):
            current_frame = frame_idx
            frame_idx += stride
            if current_frame < start_frame:
                continue
            if end_frame is not None and current_frame > end_frame:
                break

            annotated = result.plot()
            if display and not show_frame(window_name, annotated):
                break
            logger.add(result, current_frame, fps)
    finally:
        if display:
            close_window(window_name)
    logger.flush()


def run_detection_mode(
    model: YOLO,
    source: Path,
    stride: int,
    display: bool,
    logger: DetectionLogger,
    fps: float,
    start_frame: int,
    end_frame: Optional[int],
) -> None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    if start_frame:
        seek_to_frame(cap, start_frame)

    window_name = "YOLO detections"
    frame_idx = start_frame
    try:
        while True:
            if end_frame is not None and frame_idx > end_frame:
                break

            ret, frame = cap.read()
            if not ret:
                break
            current_frame = frame_idx

            results = model.predict(frame, verbose=False)
            result = results[0]
            annotated = result.plot()

            logger.add(result, current_frame, fps)

            if display and not show_frame(window_name, annotated):
                break

            frame_idx += stride
            if stride > 1:
                seek_to_frame(cap, frame_idx)
    finally:
        cap.release()
        if display:
            close_window(window_name)
    logger.flush()

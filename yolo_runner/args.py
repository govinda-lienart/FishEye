from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_SOURCE = Path("videos/first_hour.mp4.webm")
DEFAULT_WEIGHTS = Path("runs/detect/train/weights/best.pt")
DEFAULT_CONFIDENCE = 0.5
DEFAULT_LOG_DIR = Path("dataset/outputs/logs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLOv8 detections over an entire video, optionally track objects, "
            "and log detections to Parquet."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to a video file or RTSP URL.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Checkpoint to load (e.g. runs/detect/train*/weights/best.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Confidence threshold for keeping detections.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show the annotated frames in a window (press q to stop).",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=0.0,
        help="Skip everything before this timestamp (in seconds).",
    )
    parser.add_argument(
        "--end-seconds",
        type=float,
        default=None,
        help="Stop once this timestamp (seconds) is reached.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Run inference every N frames (use 30 for ~1 FPS on 30 FPS video).",
    )
    parser.add_argument(
        "--tracker",
        action="store_true",
        help="Enable ByteTrack multi-object tracking (requires --display to view).",
    )
    parser.add_argument(
        "--log-parquet",
        type=Path,
        nargs="?",
        const=DEFAULT_LOG_DIR,
        default=None,
        help=(
            "Store detection/track records as Parquet. "
            "Omit a path to drop files under dataset/outputs/logs/."
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=0,
        help="Print progress every N processed frames (0 disables progress logs).",
    )
    return parser.parse_args()

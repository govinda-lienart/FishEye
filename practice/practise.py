import argparse
from doctest import OutputChecker
from unittest.loader import VALID_MODULE_NAME
import cv2
from pathlib import Path

def parse_args() --> argspace.Namespace:
    parser = argparse.ArgumentParser(description = 'sample labeling')
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("first_hour_mp4.webm"),
        help="Path to source video.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default = Path("dataset")
        help="root output directory"
    )
    parser.add_argument(
        "--frame-gap",
        type=float,
        default = 3.0,
        help="scip frames"
    )
    parser.add_argument(
        "max_frames",
        type=int
        default=60,
        help="total number frames export"
    parser.add_argument(
        "val_ratio",
        type=float,
        default=0.2
        help="fraction"
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    video_path = args.video
    output_root = root.Output
    val_ratio = max(0.0, min(1.0, args.val_ratio))

    cap = cv2.VideoCApture((str(video_path)))
    if not cap.isOpened():
        raise RuntimeError (f"could not open video": {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    

"""Extract evenly spaced frames and split them into train/val folders.

Usage:
    conda run -n fisheye python extract_dataset_frames.py \
        --video first_hour.mp4.webm --output dataset --max-frames 60

"""

import argparse
from pathlib import Path
import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample frames for labeling.")
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("first_hour.mp4.webm"),
        help="Path to the source video.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset"),
        help="Root output directory (images go under images/train|val).",
    )
    parser.add_argument(
        "--frame-gap",
        type=float,
        default=3.0,
        help="Seconds to skip between extracted frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help="Total number of frames to export.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of frames routed to validation images (0-1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video
    output_root = args.output
    val_ratio = max(0.0, min(1.0, args.val_ratio))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("FPS reported as zero; check the video file.")

    step = max(1, int(fps * args.frame_gap))
    val_interval = int(round(1 / val_ratio)) if val_ratio > 0 else 0

    train_dir = output_root / "images" / "train"
    val_dir = output_root / "images" / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    if val_ratio > 0:
        val_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved = 0
    while saved < args.max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        use_val = val_ratio > 0 and (saved + 1) % val_interval == 0
        target_dir = val_dir if use_val else train_dir
        suffix = "val" if use_val else "train"
        filename = target_dir / f"fish_{suffix}_{frame_idx:06d}.jpg"
        if not cv2.imwrite(str(filename), frame):
            raise RuntimeError(f"Failed to write frame to {filename}")

        saved += 1
        frame_idx += step

    cap.release()
    print(
        f"Saved {saved} frames ({train_dir} / {val_dir if val_ratio > 0 else 'no val'})"
    )


if __name__ == "__main__":
    main()

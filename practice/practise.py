"""Extract evenly spaced frames and split them into train/val folders.

Exampl Usage:
   python extract_dataset_frames.py \
    --video videos/first_hour.mp4.webm \
    --output dataset \
    --frame-gap 3 \
    --max-frames 100 \
    --val-ratio 0.2

"""

import argparse
import logging
from pathlib import Path
import cv2


logging.basicConfig(format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_cli_args() -> argparse.Namespace:
    """Command-line argument parsing"""

  
    parser = argparse.ArgumentParser(description="Sample frames for labeling.")     # creation/instantiation of a ArgumentParser object stored in parser which will contain all the arguments (python extract_dataset_frames.py --help)   # it's like creating a survey form

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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point that parses arguments, samples frames, and writes them to train/val folders."""
    # --- CLI inputs & validation ---
    args = parse_cli_args() # this first step - it builts the ArgumentParser object and return the parsed (processed) Namespace with all its attribuutes (default /user)
    LOGGER.setLevel(logging.INFO if args.verbose else logging.WARNING)
    video_path = args.video
    output_root = args.output
    val_ratio = max(0.0, min(1.0, args.val_ratio))
    LOGGER.info(
        "Config parsed | video=%s output=%s frame_gap=%ss max_frames=%s val_ratio=%.2f",
        video_path,
        output_root,
        args.frame_gap,
        args.max_frames,
        val_ratio,
    )

if __name__ == "__main__":
    main() # so when using the CLI - main() function is triggered first

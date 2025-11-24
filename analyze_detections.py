from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Parquet detection logs and report fish coverage."
    )
    parser.add_argument(
        "parquet",
        type=Path,
        help="Path to the Parquet file produced by run_video.py --log-parquet.",
    )
    parser.add_argument(
        "--min-fish",
        type=int,
        default=3,
        help="How many unique fish must appear in a frame to count as 'complete'.",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="",
        default=None,
        help="Save a bar chart; optionally pass a path or directory for the PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.parquet.exists():
        raise FileNotFoundError(f"Parquet file not found: {args.parquet}")

    df = pd.read_parquet(args.parquet)
    if df.empty:
        print("No detections found in the Parquet file.")
        return

    has_tracks = df["track_id"].notna().any()
    column = "track_id" if has_tracks else "class_id"
    counts = df.groupby("frame")[column].nunique()
    total_frames = counts.index.size
    complete_frames = counts[counts >= args.min_fish].index.size
    coverage = (complete_frames / total_frames) * 100 if total_frames else 0.0

    print(f"Frames analyzed: {total_frames}")
    print(
        f"Frames with ≥{args.min_fish} unique "
        f"{'tracks' if has_tracks else 'classes'}: {complete_frames} ({coverage:.2f}%)"
    )

    freq = counts.value_counts().sort_index()
    print("\nFrames per unique-fish count:")
    for num, frames in freq.items():
        pct = (frames / total_frames) * 100 if total_frames else 0.0
        print(f"{int(num)} fish: {frames} frames ({pct:.2f}%)")

    if args.plot is not None:
        plot_arg = Path(args.plot) if args.plot else None
        output = resolve_plot_path(plot_arg, args.parquet)
        plot_fish_frequency(freq, output)


def plot_fish_frequency(freq: pd.Series, output: Path) -> None:
    """Save a bar chart showing how many frames had 1,2,3... fish detected."""
    output.parent.mkdir(parents=True, exist_ok=True)
    ax = freq.plot(kind="bar", color="#1f77b4")
    ax.set_xlabel("Unique fish per frame")
    ax.set_ylabel("Frame count")
    ax.set_title("Fish detections per frame")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    print(f"Saved bar chart to {output}")


TIMESTAMP_RE = re.compile(r"(20\d{2}[01]\d[0-3]\d_[0-2]\d[0-5]\d[0-5]\d)")


def resolve_plot_path(arg: Optional[Path], parquet_path: Path) -> Path:
    """Return a timestamped plot path, matching the Parquet file if possible."""
    timestamp = extract_timestamp(parquet_path)
    if arg is None:
        directory = parquet_path.parent
        return directory / f"fish_counts_{timestamp}.png"
    if arg.suffix:
        arg.parent.mkdir(parents=True, exist_ok=True)
        return arg
    directory = arg
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"fish_counts_{timestamp}.png"


def extract_timestamp(path: Path) -> str:
    """Derive timestamp from filename or fall back to current time."""
    match = TIMESTAMP_RE.search(path.stem)
    if match:
        return match.group(1)
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    main()

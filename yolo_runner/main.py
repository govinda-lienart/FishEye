from pathlib import Path
from ultralytics import YOLO
from .args import parse_args
from .detection import run_detection_mode, run_tracker_mode
from .records import DetectionLogger
from .video_utils import compute_frame_bounds, read_fps


def run_cli() -> None:
    args = parse_args()
    run(args)


def run(args):
    source: Path = args.source
    weights: Path = args.weights

    if not source.exists():
        raise FileNotFoundError(f"Video source does not exist: {source}")
    if not weights.exists():
        raise FileNotFoundError(f"Missing model weights: {weights}")

    model = YOLO(str(weights))
    fps = read_fps(source)
    start_frame, end_frame = compute_frame_bounds(
        fps, args.start_seconds, args.end_seconds
    )
    logger = DetectionLogger(args.log_parquet, args.progress_interval)

    if args.tracker:
        log_path = run_tracker_mode(
            model=model,
            source=source,
            stride=args.stride,
            display=args.display,
            logger=logger,
            fps=fps,
            start_frame=start_frame,
            end_frame=end_frame,
        )
    else:
        log_path = run_detection_mode(
            model=model,
            source=source,
            stride=args.stride,
            display=args.display,
            logger=logger,
            fps=fps,
            start_frame=start_frame,
            end_frame=end_frame,
        )
    return logger.log_path

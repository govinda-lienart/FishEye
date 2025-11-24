from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from ultralytics import YOLO

DEFAULT_SOURCE = Path("videos/first_hour.mp4.webm")
DEFAULT_WEIGHTS = Path("runs/detect/train/weights/best.pt")
DEFAULT_CONFIDENCE = 0.5
TRACKER_CONFIG = "ultralytics/cfg/trackers/bytetrack.yaml"
DEFAULT_LOG_DIR = Path("dataset/outputs/logs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 detections over an entire video and display the overlay."
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


def build_records(
    result,
    frame_idx: int,
    fps: float,
) -> List[Dict[str, Any]]:
    """Convert a YOLO Result into serializable detection records."""
    boxes = result.boxes
    if boxes is None or boxes.data.shape[0] == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    ids: Optional[List[Optional[int]]]
    if boxes.id is not None:
        ids = boxes.id.cpu().numpy().astype(int).tolist()
    else:
        ids = [None] * len(xyxy)

    timestamp = frame_idx / fps if fps else None
    records: List[Dict[str, Any]] = []
    for (x1, y1, x2, y2), conf, cls_id, track_id in zip(xyxy, confs, classes, ids):
        records.append(
            {
                "frame": frame_idx,
                "timestamp": timestamp,
                "track_id": track_id,
                "class_id": int(cls_id),
                "confidence": float(conf),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )
    return records


def write_parquet(records: List[Dict[str, Any]], path: Path) -> None:
    """Persist detection records to Parquet."""
    if not records:
        print("No detections recorded; skipping Parquet write.")
        return
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Install pandas (and pyarrow/fastparquet) to use --log-parquet."
        ) from exc

    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = DEFAULT_LOG_DIR / f"detections_{timestamp}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(path)
    print(f"Wrote {len(df)} detections to {path}")


def resolve_log_path(path: Optional[Path]) -> Optional[Path]:
    """Turn a user argument into a concrete Parquet file path."""
    if path is None:
        return None
    if path.suffix.lower() == ".parquet":
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (path or DEFAULT_LOG_DIR) / f"detections_{timestamp}.parquet"


def maybe_print_progress(frame_idx: int, records_count: int, interval: int) -> None:
    """Optionally print progress while logging detections."""
    if interval > 0 and frame_idx > 0 and frame_idx % interval == 0:
        print(f"Processed frame {frame_idx} (total detections logged: {records_count})")


def compute_frame_bounds(
    fps: float, start_seconds: float, end_seconds: Optional[float]
) -> tuple[int, Optional[int]]:
    """Convert requested timestamps to frame indices."""
    if start_seconds < 0:
        raise ValueError("--start-seconds cannot be negative.")
    if end_seconds is not None and end_seconds <= start_seconds:
        raise ValueError("--end-seconds must be greater than --start-seconds.")
    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps) if end_seconds is not None else None
    return start_frame, end_frame


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"Video source does not exist: {args.source}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Missing model weights: {args.weights}")

    model = YOLO(str(args.weights))

    cap_for_meta = cv2.VideoCapture(str(args.source))
    if not cap_for_meta.isOpened():
        raise RuntimeError(f"Could not open video: {args.source}")
    fps = cap_for_meta.get(cv2.CAP_PROP_FPS) or 30.0
    cap_for_meta.release()
    start_frame, end_frame = compute_frame_bounds(
        fps, args.start_seconds, args.end_seconds
    )

    log_path = resolve_log_path(args.log_parquet)
    records: List[Dict[str, Any]] = []

    if args.tracker:
        window_name = "YOLO ByteTrack"
        frame_idx = 0
        try:
            for result in model.track(
                source=str(args.source),
                conf=args.conf,
                tracker=TRACKER_CONFIG,
                vid_stride=args.stride,
                stream=True,
                show=False,
                save=False,
                verbose=False,
                persist=True,
            ):
                current_frame = frame_idx
                frame_idx += args.stride
                if current_frame < start_frame:
                    continue
                if end_frame is not None and current_frame > end_frame:
                    break

                annotated = result.plot()
                if args.display:
                    cv2.imshow(window_name, annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if log_path:
                    records.extend(build_records(result, current_frame, fps))
                    maybe_print_progress(
                        current_frame, len(records), args.progress_interval
                    )
        finally:
            if args.display:
                cv2.destroyWindow(window_name)
        if log_path:
            write_parquet(records, log_path)
        return

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.source}")

    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

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

            results = model.predict(
                frame,
                conf=args.conf,
                verbose=False,
            )
            result = results[0]
            annotated = result.plot()
            if log_path:
                records.extend(build_records(result, current_frame, fps))
                maybe_print_progress(
                    current_frame, len(records), args.progress_interval
                )

            if args.display:
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += args.stride
            if args.stride > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    finally:
        cap.release()
        if args.display:
            cv2.destroyWindow(window_name)
    if log_path:
        write_parquet(records, log_path)


if __name__ == "__main__":
    main()

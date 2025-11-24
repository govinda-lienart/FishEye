from __future__ import annotations

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

DEFAULT_SOURCE = Path("videos/first_hour.mp4.webm")
DEFAULT_WEIGHTS = Path("runs/detect/train/weights/best.pt")
DEFAULT_CONFIDENCE = 0.5
TRACKER_CONFIG = "ultralytics/cfg/trackers/bytetrack.yaml"


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"Video source does not exist: {args.source}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Missing model weights: {args.weights}")

    model = YOLO(str(args.weights))

    if args.tracker:
        model.track(
            source=str(args.source),
            conf=args.conf,
            tracker=TRACKER_CONFIG,
            vid_stride=args.stride,
            show=args.display,
            stream=False,
            save=False,
        )
        return

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.source}")

    window_name = "YOLO detections"
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            run_inference = frame_idx % args.stride == 0
            if run_inference or "annotated" not in locals():
                results = model.predict(
                    frame,
                    conf=args.conf,
                    verbose=False,
                )
                annotated = results[0].plot()
            else:
                # Reuse the last annotated frame to keep display smooth.
                annotated = annotated.copy()

            if args.display:
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1
    finally:
        cap.release()
        if args.display:
            cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()

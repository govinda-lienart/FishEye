from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

DEFAULT_SOURCE = Path("videos/first_hour.mp4.webm")
DEFAULT_WEIGHTS = Path("yolov8n.pt")
DEFAULT_OUTPUT = Path("frame_with_boxes.jpg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLOv8 on a single image or the first frame of a video and "
            "save an annotated copy."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to an image or video. Videos use the first frame.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Checkpoint to load (e.g. runs/detect/train/weights/best.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to store the annotated frame/image.",
    )
    return parser.parse_args()


def load_frame(source: Path) -> "cv2.typing.MatLike":
    """Return an image array, reading from an image or video source."""
    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    image = cv2.imread(str(source))
    if image is not None:
        return image

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read first frame from video")
    return frame


def main() -> None:
    args = parse_args()

    frame = load_frame(args.source)
    model = YOLO(str(args.weights))
    results = model(frame)

    for box in results[0].boxes:
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        print(f"class {cls} conf {conf:.2f} box {xyxy}")

    annotated = results[0].plot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), annotated)


if __name__ == "__main__":
    main()

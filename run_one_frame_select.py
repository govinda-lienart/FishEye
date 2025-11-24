from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO on a selected image or video frame.")
    parser.add_argument("--source", type=Path, required=True, help="Path to an image or video.")
    parser.add_argument("--weights", type=Path, required=True, help="Checkpoint to load.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("frame_with_boxes.jpg"),
        help="Destination for the annotated frame.",
    )
    return parser.parse_args()


def load_frame(source: Path) -> "cv2.typing.MatLike":
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    annotated = results[0].plot()
    if args.output.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        args.output = args.output.with_suffix(".jpg")
    cv2.imwrite(str(args.output), annotated)
    print(f"Annotated frame saved to {args.output}")


if __name__ == "__main__":
    main()

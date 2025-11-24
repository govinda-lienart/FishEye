from pathlib import Path

from ultralytics import YOLO

DATA_CONFIG = Path("dataset/fish.yaml")
MODEL_WEIGHTS = Path("yolov8n.pt")
EPOCHS = 50
IMAGE_SIZE = 640


def main() -> None:
    """Train YOLOv8 using the prepared dataset split."""
    if not DATA_CONFIG.exists():
        raise FileNotFoundError(f"Missing data config: {DATA_CONFIG}")

    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Missing base weights: {MODEL_WEIGHTS}. "
            "Place yolov8n.pt (or your chosen checkpoint) in the repository root."
        )

    model = YOLO(str(MODEL_WEIGHTS))
    model.train(data=str(DATA_CONFIG), epochs=EPOCHS, imgsz=IMAGE_SIZE)


if __name__ == "__main__":
    main()

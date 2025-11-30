from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

DATA_CONFIG = Path("dataset/fish.yaml")
MODEL_WEIGHTS = Path("yolov8n.pt")
EPOCHS = 50
IMAGE_SIZE = 640
RUNS_DIR = Path("runs/detect")


def main() -> None:
    if not DATA_CONFIG.exists():
        raise FileNotFoundError(f"Missing data config: {DATA_CONFIG}")

    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Missing base weights: {MODEL_WEIGHTS}. "
            "Place yolov8n.pt (or your chosen checkpoint) in the repository root."
        )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("train_%Y%m%d_%H%M%S")

    model = YOLO(str(MODEL_WEIGHTS))
    model.train(
        data=str(DATA_CONFIG),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        project=str(RUNS_DIR),
        name=run_name,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_LOG_DIR = Path("dataset/outputs/logs")


@dataclass
class DetectionLogger:
    log_path: Optional[Path]
    progress_interval: int = 0
    records: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.log_path is not None and self.log_path.suffix.lower() != ".parquet":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = (self.log_path or DEFAULT_LOG_DIR) / f"detections_{timestamp}.parquet"
        elif self.log_path is None:
            self.log_path = None

    @property
    def enabled(self) -> bool:
        return self.log_path is not None

    def add(self, result, frame_idx: int, fps: float) -> None:
        if not self.enabled:
            return
        self.records.extend(build_records(result, frame_idx, fps))
        self._maybe_print(frame_idx)

    def _maybe_print(self, frame_idx: int) -> None:
        if (
            self.progress_interval > 0
            and frame_idx > 0
            and frame_idx % self.progress_interval == 0
        ):
            print(
                f"Processed frame {frame_idx} (total detections logged: {len(self.records)})"
            )

    def flush(self) -> None:
        if not self.enabled:
            return
        if not self.records:
            print("No detections recorded; skipping Parquet write.")
            return
        assert self.log_path is not None
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Install pandas (and pyarrow/fastparquet) to use --log-parquet."
            ) from exc
        df = pd.DataFrame(self.records)
        df.to_parquet(self.log_path)
        print(f"Wrote {len(df)} detections to {self.log_path}")


def build_records(result, frame_idx: int, fps: float) -> List[Dict[str, Any]]:
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

import logging
from argparse import Namespace
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from yolo_runner.args import DEFAULT_LOG_DIR, DEFAULT_SOURCE, DEFAULT_WEIGHTS
from yolo_runner.main import run as run_yolo

app = FastAPI()
templates = Jinja2Templates(directory="templates")
logger = logging.getLogger("uvicorn.error")


def _str_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    return Path(value).expanduser()


def _float(value: str | None, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _optional_float(value: str | None) -> Optional[float]:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    defaults = {
        "source": str(DEFAULT_SOURCE),
        "weights": str(DEFAULT_WEIGHTS),
        "start_seconds": "0",
        "end_seconds": "",
        "stride": "1",
        "tracker": False,
        "display": False,
        "log_enabled": False,
        "log_path": "",
        "progress_interval": "0",
    }
    return templates.TemplateResponse(
        "index.html", {"request": request, "message": None, "defaults": defaults}
    )


@app.post("/", response_class=HTMLResponse)
async def run(
    request: Request,
    source: str = Form(default=str(DEFAULT_SOURCE)),
    weights: str = Form(default=str(DEFAULT_WEIGHTS)),
    start_seconds: str = Form(default="0"),
    end_seconds: str = Form(default=""),
    stride: str = Form(default="1"),
    tracker: Optional[str] = Form(default=None),
    display: Optional[str] = Form(default=None),
    log_enabled: Optional[str] = Form(default=None),
    log_path: str = Form(default=""),
    progress_interval: str = Form(default="0"),
):
    source_path = _str_path(source, DEFAULT_SOURCE)
    weights_path = _str_path(weights, DEFAULT_WEIGHTS)
    start_value = _float(start_seconds, 0.0)
    end_value = _optional_float(end_seconds)
    stride_value = max(1, _int(stride, 1))
    progress_value = max(0, _int(progress_interval, 0))
    tracker_enabled = tracker is not None
    display_enabled = display is not None
    log_enabled_flag = log_enabled is not None
    log_parquet = (
        _str_path(log_path, DEFAULT_LOG_DIR)
        if log_enabled_flag and log_path.strip()
        else (DEFAULT_LOG_DIR if log_enabled_flag else None)
    )

    args = Namespace(
        source=source_path,
        weights=weights_path,
        display=display_enabled,
        start_seconds=start_value,
        end_seconds=end_value,
        stride=stride_value,
        tracker=tracker_enabled,
        log_parquet=log_parquet,
        progress_interval=progress_value,
    )

    logger.info(
        "Running YOLO: source=%s weights=%s stride=%s tracker=%s display=%s "
        "start=%.2f end=%s log=%s progress=%s",
        source_path,
        weights_path,
        stride_value,
        tracker_enabled,
        display_enabled,
        start_value,
        end_value,
        log_parquet or "disabled",
        progress_value,
    )

    message: str
    try:
        log_file = run_yolo(args)
        if log_file:
            message = f"Run completed. Log saved to {log_file}"
            logger.info("Run finished. Log saved to %s", log_file)
        else:
            message = "Run completed (logging disabled)."
            logger.info("Run finished (logging disabled).")
    except Exception as exc:  # pragma: no cover - runtime path
        logger.exception("Run failed")
        message = f"Error: {exc}"

    defaults = {
        "source": source,
        "weights": weights,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "stride": stride,
        "tracker": tracker_enabled,
        "display": display_enabled,
        "log_enabled": log_enabled_flag,
        "log_path": log_path,
        "progress_interval": progress_interval,
    }
    return templates.TemplateResponse(
        "index.html", {"request": request, "message": message, "defaults": defaults}
    )

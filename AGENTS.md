# Repository Guidelines

## Project Structure & Module Organization
Keep the root focused on three fast-iteration scripts: `practise.py` for short playback checks, `run_one_frame.py` for YOLOv8 spot tests, and `' check_video_io.py'` for FPS sanity checks. Store bulky captures inside `videos/` and derived clips or frames under `dataset/` (create subfolders like `frames/` or `outputs/` as needed) so the repository stays tidy. Treat `yolov8n.pt` and any additional weights as read-only artifacts; if you need variants, save them in `models/` rather than overwriting the defaults. Documentation and coordination notes belong in `context.md` or this guide.

## Build, Test, and Development Commands
Create an isolated environment before running experiments:
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip opencv-python ultralytics numpy
```
Use `python ' check_video_io.py'` to confirm OpenCV can read `videos/first_hour.mp4.webm`. Run `python practise.py` to manually inspect the first few seconds, and `python run_one_frame.py` to generate a YOLO-annotated still (outputs land near the script unless you override the path). When tests are added, execute `pytest tests/` from the repo root; keep data fixtures lightweight so the suite stays fast.

## Coding Style & Naming Conventions
Target Python 3.10+, 4-space indentation, and `snake_case` for functions, variables, and filenames. Group shared constants (paths, FPS, model names) at the top of each script with uppercase identifiers. Whenever code grows beyond a prototype, factor helpers into dedicated modules (e.g., `video_io/reader.py`) and add type hints to public functions. Before sharing work, run `ruff --fix .` followed by `black .` to keep formatting consistent.

## Testing Guidelines
Mirror the module layout inside `tests/` (for example, `tests/test_video_io.py`). Fake `cv2.VideoCapture` or point to trimmed clips under `dataset/` to avoid loading the full `first_hour` asset on every run. Aim for deterministic counts (frame totals, detection tallies) so regressions are easy to spot. When new behaviors rely on external services, capture representative payloads as JSON fixtures checked into `tests/fixtures/`.

## Commit & Pull Request Guidelines
Adopt Conventional Commit messages (`feat: add frame extractor`). Each pull request should describe the scenario exercised (commands run, FPS observed, sample detection counts) and attach screenshots or saved frames when visuals change. Call out new dependencies and mention if large files were copied into `dataset/` or `videos/` so reviewers can reproduce the environment quickly.

## Security & Configuration Tips
Treat `videos/first_hour.mp4.webm` and any derived media as confidential—copy them before trimming or annotating, never overwrite in place. Keep credentials for future database work inside an untracked `.env`, and record DSN details in `context.md` instead of hardcoding them in scripts. Avoid committing temporary notebooks or raw outputs at the repo root; place exploratory assets under `notes/` or `outputs/` folders that can be ignored later.

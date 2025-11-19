# Fish Behavior Study Context

## Step 1: Tracking Pipeline
Convert video frames into structured data. OpenCV reads frames (e.g., 2 fps) and passes them to a pretrained YOLO detector to locate each fish. Detections feed into Deep SORT or ByteTrack so every fish keeps a stable ID over time, even when paths cross. For each frame compute metrics such as timestamp, fish ID, center x/y, velocity components, orientation proxies, and distances to decorations (gravel, plants, Moai statue). Insert these facts into MySQL with indexes on timestamp and fish ID so any time slice can be queried instantly without reprocessing the raw footage.

## Step 2: Unsupervised Exploration
Once trajectories live in MySQL, pull short windows (5–10 seconds per fish) into Pandas. Engineer features like mean/variance of speed, time spent near the substrate (based on y-position thresholds), vertical acceleration spikes, and curvature of the path. Use scikit-learn to run PCA/UMAP for dimensionality reduction, then cluster the windows (k-means, DBSCAN, or HDBSCAN). Plot representative trajectories per cluster with Matplotlib/Plotly to interpret natural behavior groups—hovering near gravel, darting through open water, plant exploration—before committing to labels.

## Step 3: Supervised Behavior Recognition
Label segments where feeding occurs by watching the video and recording start/end timestamps per fish; store them in a `behavior_labels` table linked to the detection rows. Build a PyTorch Dataset that queries MySQL for each labeled window, stacks the time-series metrics into tensors, and returns `(sequence, label)` pairs. Train an LSTM or transformer (with dropout/batch norm as needed) using BCE loss and Adam, validate on held-out windows, and save the best checkpoint. At runtime, stream new tracking data into MySQL, fetch the latest window, run the model in eval mode, and log feeding predictions plus confidences so any section—live or recorded—gets an immediate behavior verdict.

## Core Modules and Their Roles
- OpenCV (`opencv-python`): grabs frames and performs basic resizing/normalization before detection.
- Ultralytics YOLO / TorchVision models: supply pretrained detectors that find fish without heavy training.
- Deep SORT / ByteTrack: maintain consistent fish identities, powering clean trajectories for downstream analysis.
- NumPy: handles vectorized math when converting boxes into positions, speeds, and accelerations.
- MySQL (with optional SQLAlchemy/Pandas): stores detections and metrics centrally with indexes for fast window queries and long-term durability.
- scikit-learn: provides PCA/UMAP and clustering (k-means/DBSCAN/HDBSCAN) for unsupervised behavior discovery.
- PyTorch: implements the supervised behavior classifier (LSTM/transformer) that maps trajectories to feeding labels.
- Matplotlib / Plotly: visualize trajectories, clusters, and model predictions for debugging and reporting.

## Storage Strategy Notes
Follow the workflow recommended by your friend: run YOLO + tracker, write one row per fish per frame (timestamp, fish ID, x/y, speed, orientation, distances) into MySQL with composite indexes on timestamp/fish_id. Query short trajectory windows for both unsupervised clustering and supervised PyTorch training; this keeps everything fast, local, and reproducible without big-data infrastructure. If datasets someday grow to hundreds of hours, MySQL can scale to a proper server or be migrated to PostgreSQL/cloud without redesigning the pipeline.

## Subproject 1: Tracking Recorder (Key Steps)
Frame reader: use OpenCV to sample frames at a manageable rate (e.g., 2 fps) and ensure consistent resolution. Detector integration: load a pretrained YOLO model, tune confidence/NMS thresholds, and emit bounding boxes per frame. Tracking IDs: feed detections into Deep SORT or ByteTrack to get persistent fish IDs, configuring embedding models/Kalman filters so ID switches stay rare. Metric logging: compute derived metrics (center x/y, speed components, distance to gravel/objects) and batch-insert rows into MySQL with indexes on timestamp plus fish ID. Recorder script: wrap the loop into a script/service that processes stored video or live feed, reports FPS, handles dropped frames, and can resume where it left off.

## Subproject 2: Unsupervised Exploration (Key Steps)
Data access layer: implement helper functions that query MySQL for a fish/time interval and return Pandas DataFrames. Feature engineering: compute per-window stats (mean/variance of speed, dwell time near substrate, vertical acceleration, path curvature). Dimensionality reduction: run PCA or UMAP on the feature matrix to visualize structure. Clustering: apply k-means/DBSCAN/HDBSCAN to the reduced vectors and annotate each window with a cluster ID. Visualization: plot representative trajectories and feature summaries per cluster with Matplotlib/Plotly to interpret the behaviors and decide which ones resemble feeding.

## Subproject 3: Supervised Recognizer (Key Steps)
Labeling workflow: watch video segments, record feeding intervals per fish, and store them in a `behavior_labels` table keyed by timestamp. Dataset builder: create a PyTorch Dataset that queries labeled windows, stacks the time-series metrics into tensors, and yields `(sequence, label)` pairs. Model architecture: implement an LSTM/transformer encoder with a final linear layer for binary feeding classification plus optional dropout/batch norm. Training loop: train/validate with Adam + BCE loss, log metrics, and checkpoint the best weights. Runtime inference: deploy a loop that pulls the latest window from MySQL, runs `model.eval()` to predict feeding in real time, and logs or displays the predictions for dashboards and further analysis.


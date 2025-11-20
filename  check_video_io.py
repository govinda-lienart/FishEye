import cv2

cap = cv2.VideoCapture("first_hour.mp4.webm")
if not cap.isOpened():
    raise RuntimeError("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps:.2f}")

max_time = 5.0
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    timestamp = frame_idx / fps
    if timestamp > max_time:
        break
    print(f"Frame {frame_idx} at {timestamp:.2f}s")
    frame_idx += 1


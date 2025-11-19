import cv2

# instantiate object

cap = cv2.VideoCapture( "first_hour.mp4.webm")
if not cap.isOpened():
    raise RuntimeError('could not open video')

fps = cap.get(cv2.CAP_PROP_FPS)

frame_index = 0
fps_max_duration = 5.0

# while loop

while True:
    ret, frame = cap.read()
    if not ret:
        break
    timestamp = frame_index / fps
    if timestamp > fps_max_duration:
        break
    print (f"Frame {frame_index} checked at time {timestamp}")
    frame_index += 1



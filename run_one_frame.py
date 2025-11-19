from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture("first_hour.mp4.webm")

ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("Could not read first frame")

model = YOLO("yolov8n.pt")            # tiny pretrained YOLO model
results = model(frame)                # run detection on the first
frame

for box in results[0].boxes:
    xyxy = box.xyxy[0].tolist()
    conf = float(box.conf[0])
    cls = int(box.cls[0])
    print(f"class {cls} conf {conf:.2f} box {xyxy}")

annotated = results[0].plot()         # draw boxes/labels on the frame
cv2.imwrite("frame_with_boxes.jpg", annotated)
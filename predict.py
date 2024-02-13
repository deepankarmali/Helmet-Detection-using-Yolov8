import os
from ultralytics import YOLO
import cv2

video_path = r'videos/production_id_5052423 (1080p).mp4'
video_path_out = '{}_out2.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file.")
    exit()

ret, frame = cap.read()

if not ret:
    print("Error reading the first frame.")
    exit()

H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'best.pt')

model = YOLO(model_path)
model.to('cuda')

threshold = 0.5

while ret:
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    cv2.imshow('Processed Video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()

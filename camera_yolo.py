import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO

RTSP_URL = "rtsp://service:Praktik26!@172.16.1.25:554/"

model = YOLO("yolov8n.pt")

frame = None
running = True
last_frame_time = time.time()


def create_error_frame(message, code="CAM01"):
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    cv2.putText(img, f"Error {code}", (150, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(img, message, (120, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img


# ⭐ Kamera läses i egen tråd
def camera_reader():
    global frame, running, last_frame_time

    cap = cv2.VideoCapture(RTSP_URL)

    while running:
        ret, new_frame = cap.read()

        if ret:
            frame = new_frame
            last_frame_time = time.time()
        else:
            time.sleep(0.1)

    cap.release()


threading.Thread(target=camera_reader, daemon=True).start()

TIMEOUT = 3
frame_count = 0
last_annotated_frame = None


while True:

    # ⭐ Om kameran inte skickat bild på 3 sek → errorbild
    if time.time() - last_frame_time > TIMEOUT:
        error_img = create_error_frame("Camera connection lost")
        cv2.imshow("YOLO Kamera", error_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue

    if frame is None:
        continue

    frame_count += 1

    if frame_count % 3 == 0:
        results = model(frame, verbose=False)
        last_annotated_frame = results[0].plot()

    if last_annotated_frame is not None:
        cv2.imshow("YOLO Kamera", last_annotated_frame)
    else:
        cv2.imshow("YOLO Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


running = False
cv2.destroyAllWindows()

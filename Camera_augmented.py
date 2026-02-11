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

CONFIDENCE_THRESHOLD = 0.40

# ============================================================
# FILTER - Välj vilka objekt som ska visas
# ============================================================
# Lägg till de objekt ni vill se i listan
# Exempel: ["person"] = bara personer
# Exempel: ["person", "car"] = personer och bilar
# Exempel: [] = tom lista = visa ALLT

ALLOWED_CLASSES = ["person"]

COLORS = {
    "person": (0, 255, 0),
    "car": (0, 0, 255),
    "truck": (0, 0, 200),
    "chair": (255, 165, 0),
    "cell phone": (255, 0, 255),
    "laptop": (255, 255, 0),
}
DEFAULT_COLOR = (200, 200, 200)


def create_error_frame(message, code="CAM01"):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, f"Error {code}", (150, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, message, (120, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


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


# ============================================================
# SPARA DETEKTIONER - Lösningen för flimmer
# ============================================================
# Istället för att spara en färdig bild sparar vi DATAN
# (koordinater, namn, confidence) och ritar om den varje frame

last_detections = []


def extract_detections(results):
    """Hämtar ut detektionsdata från YOLO-resultaten."""
    detections = []

    boxes = results[0].boxes

    for box in boxes:
        confidence = float(box.conf[0])

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # FILTRERA - Hoppa över objekt som inte är i listan
        # Om listan är tom visas allt
        if ALLOWED_CLASSES and class_name not in ALLOWED_CLASSES:
            continue

        detections.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "class_name": class_name,
            "confidence": confidence
        })

    return detections


def draw_detections(img, detections):
    """Ritar sparade detektioner på en bild."""

    for det in detections:
        color = COLORS.get(det["class_name"], DEFAULT_COLOR)

        # Rita rektangel
        cv2.rectangle(img, (det["x1"], det["y1"]),
                       (det["x2"], det["y2"]), color, 2)

        # Skapa label
        label = f"{det['class_name']} {det['confidence']:.0%}"

        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # Bakgrund för text
        cv2.rectangle(
            img,
            (det["x1"], det["y1"] - text_height - 10),
            (det["x1"] + text_width, det["y1"]),
            color, -1
        )

        # Text
        cv2.putText(img, label, (det["x1"], det["y1"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img


# Starta kameratråden
threading.Thread(target=camera_reader, daemon=True).start()

TIMEOUT = 3
frame_count = 0

while True:

    if time.time() - last_frame_time > TIMEOUT:
        error_img = create_error_frame("Camera connection lost")
        cv2.imshow("YOLO Kamera", error_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    if frame is None:
        continue

    frame_count += 1

    # Kör YOLO var 3:e frame - spara DATAN
    if frame_count % 3 == 0:
        results = model(frame, verbose=False)
        last_detections = extract_detections(results)

    # Rita senaste detektionerna på VARJE frame
    display_frame = draw_detections(frame.copy(), last_detections)

    cv2.imshow("YOLO Kamera", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

running = False
cv2.destroyAllWindows()

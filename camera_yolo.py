import cv2
from ultralytics import YOLO

# Ladda AI-modellen
model = YOLO("yolov8n.pt")

# Koppla upp RTSP-kameran
cap = cv2.VideoCapture("rtsp://service:Praktik26!@172.16.1.25:554/")

frame_count = 0
last_annotated_frame = None   # Sparar senaste AI-resultatet

while True:

    # Läs bild från kameran
    ret, frame = cap.read()

    if not ret:
        print("Kunde inte läsa från kameran")
        break

    # Ändra storlek för bättre prestanda
    frame = cv2.resize(frame, (640, 360))

    frame_count += 1

    # Kör AI var tredje frame
    if frame_count % 3 == 0:
        results = model(frame, verbose=False)
        last_annotated_frame = results[0].plot()

    # Visa senaste AI-resultatet
    if last_annotated_frame is not None:
        cv2.imshow("YOLO Kamera", last_annotated_frame)
    else:
        cv2.imshow("YOLO Kamera", frame)

    # Tryck Q för att avsluta
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Stäng kamera och fönster
cap.release()
cv2.destroyAllWindows()

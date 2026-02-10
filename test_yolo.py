from ultralytics import YOLO

# Ladda modell
model = YOLO("yolov8n.pt")

# Kör analys
results = model("Bilder/test.jpg")

# Visa resultat
results[0].show()
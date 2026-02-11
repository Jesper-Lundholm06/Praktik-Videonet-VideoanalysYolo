import cv2
import numpy as np
import time
import os
import subprocess
import threading
from flask import Flask, render_template, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)

# ============================================================
# KAMERA & YOLO - Samma som förut
# ============================================================

RTSP_URL = "rtsp://service:Praktik26!@172.16.1.25:554/inst=2"
model = YOLO("yolov8n.pt")
model.to("cuda")

frame = None
running = True
last_frame_time = time.time()

CONFIDENCE_THRESHOLD = 0.40
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

last_detections = []


def create_error_frame(message, code="CAM01"):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, f"Error {code}", (150, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, message, (120, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img
def camera_reader():
    global frame, running, last_frame_time

    cap = None
    CONNECT_TIMEOUT = 5   # sek utan frames → reconnect

    while running:

        # ===== CONNECT =====
        if cap is None or not cap.isOpened():
            print("🔄 Försöker ansluta kamera...")

            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            time.sleep(1)

            if not cap.isOpened():
                print("❌ Kunde inte ansluta kamera")
                cap = None
                time.sleep(2)
                continue

            print("✅ Kamera ansluten")
            last_frame_time = time.time()

        # ===== READ FRAME =====
        ret, new_frame = cap.read()

        if ret:
            frame = new_frame
            last_frame_time = time.time()

        # ===== WATCHDOG =====
        if time.time() - last_frame_time > CONNECT_TIMEOUT:
            print("⚠️ Ingen frame mottagen → reconnectar kamera")

            cap.release()
            cap = None
            frame = None
            time.sleep(1)


def extract_detections(results):
    detections = []
    boxes = results[0].boxes
    for box in boxes:
        confidence = float(box.conf[0])
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        if ALLOWED_CLASSES and class_name not in ALLOWED_CLASSES:
            continue
        detections.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "class_name": class_name, "confidence": confidence
        })
    return detections


def draw_detections(img, detections):
    for det in detections:
        color = COLORS.get(det["class_name"], DEFAULT_COLOR)
        cv2.rectangle(img, (det["x1"], det["y1"]),
                       (det["x2"], det["y2"]), color, 2)
        label = f"{det['class_name']} {det['confidence']:.0%}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (det["x1"], det["y1"] - text_height - 10),
                       (det["x1"] + text_width, det["y1"]), color, -1)
        cv2.putText(img, label, (det["x1"], det["y1"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


# ============================================================
# HLS - Mapp där H.264-segmenten sparas
# ============================================================
# FFmpeg skapar små videofiler (.ts) och en spellista (.m3u8)
# Webbläsaren läser spellistan och spelar upp segmenten

HLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hls')
os.makedirs(HLS_DIR, exist_ok=True)

# Storlek på videon som skickas till FFmpeg
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# ============================================================
# FFMPEG - Kodar frames till H.264
# ============================================================
# FFmpeg tar emot råa bilder via en "pipe" (stdin)
# och skapar H.264-kodade HLS-segment
def start_ffmpeg():
    return subprocess.Popen([
        'ffmpeg',
        '-y',

        # INPUT (OpenCV skickar rå BGR)
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{FRAME_WIDTH}x{FRAME_HEIGHT}',
        '-r', '10',              # Högre FPS
        '-i', '-',

        # GPU ENCODING (NVENC)
        '-c:v', 'h264_nvenc',
        '-preset', 'p1',         # Snabbast preset
        '-tune', 'll',           # Low latency
        '-rc', 'vbr',
        '-b:v', '2M',

        # ⭐ FÄRGFIX
        '-pix_fmt', 'yuv420p',

        # Keyframes
        '-g', '10',

        # HLS OUTPUT
        '-f', 'hls',
        '-hls_time', '1',      # Mindre latency
        '-hls_list_size', '2',   # Mindre buffert
        '-hls_flags', 'delete_segments+temp_file',

        os.path.join(HLS_DIR, 'stream.m3u8')
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


# ============================================================
# PROCESSOR-TRÅD - Kör YOLO och skickar till FFmpeg
# ============================================================

def stream_processor():
    """Bearbetar frames med YOLO och skickar till FFmpeg."""
    global last_detections

    ffmpeg = start_ffmpeg()
    frame_count = 0
    TARGET_FPS = 10
    FRAME_TIME = 1.0 / TARGET_FPS
    last_send_time = time.time()


    while running:
        if frame is None:
            display = create_error_frame("Camera reconnecting...")
            ffmpeg.stdin.write(display.tobytes())
            time.sleep(0.1)
            continue
    
        current_frame = frame.copy()
        frame_count += 1
        start_total = time.time()

        # Kör YOLO var 3:e frame
        if frame_count % 1 == 0:
            start = time.time()

            results = model(current_frame, device="cuda", verbose=False)

            print("YOLO inference time:", round(time.time() - start, 3), "sek")

            last_detections = extract_detections(results)
            

        # Rita detektioner
        if time.time() - last_frame_time > 3:
            display = create_error_frame("Camera connection lost")
        else:
            display = draw_detections(current_frame, last_detections)

        # Ändra storlek till det FFmpeg förväntar sig
        display = cv2.resize(display, (FRAME_WIDTH, FRAME_HEIGHT))
       # ===== FPS pacing =====
        now = time.time()
        sleep_time = FRAME_TIME - (now - last_send_time)

        if sleep_time > 0:
            time.sleep(sleep_time)

        last_send_time = time.time()

# ===== Skicka till FFmpeg =====
        try:
            start = time.time()
            ffmpeg.stdin.write(display.tobytes())
            print("PIPE:", round(time.time() - start, 3), "sek")
            print("TOTAL:", round(time.time() - start_total, 3), "sek")
        except BrokenPipeError:
            print("FFmpeg-pipe bröts. Startar om...")
            ffmpeg = start_ffmpeg()

      
      

    ffmpeg.stdin.close()
    ffmpeg.wait()


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/hls/<path:filename>')
def hls_stream(filename):
    """Serverar HLS-filer (.m3u8 och .ts) till webbläsaren."""
    return send_from_directory(HLS_DIR, filename)


# ============================================================
# STARTA ALLT
# ============================================================

if __name__ == '__main__':
    threading.Thread(target=camera_reader, daemon=True).start()
    threading.Thread(target=stream_processor, daemon=True).start()

    print("Servern startar pa http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
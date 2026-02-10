import cv2

# RTSP-adress till er Bosch-kamera
# Byt ut användarnamn, lösenord och IP om det behövs
cap = cv2.VideoCapture("rtsp://service:Praktik26!@172.16.1.25:554/")

# Kolla om strömmen öppnades
if not cap.isOpened():
    print("FEL: Kunde inte ansluta till kameran!")
    exit()

print("Ansluten till kameran! Tryck 'q' för att stänga.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Tappade anslutningen till kameran.")
        break

    cv2.imshow("Bosch Kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


## Vad är nytt jämfört med videofilen?


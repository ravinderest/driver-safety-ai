import cv2
import time
import winsound
import os
import threading
from openai import OpenAI

# -------- SETTINGS --------
VIDEO_FILE = "driver.mp4"
MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b"
ALERT_AFTER_SECONDS = 1.0

# -------- NEBIUS CLIENT --------
client = None
api_key = os.getenv("NEBIUS_API_KEY")

if api_key:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.tokenfactory.nebius.com/v1/"
    )
else:
    print("NEBIUS_API_KEY missing. Video alert will still work.")

# -------- AI FUNCTION --------
def ask_nemotron():
    if client is None:
        print("AI advice unavailable: API key not configured.")
        return

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a driver safety assistant. "
                        "Give one short, practical safety recommendation. "
                        "Do not claim to diagnose medical conditions."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "The computer vision system detected possible "
                        "prolonged eye closure. Give a safety recommendation."
                    )
                }
            ],
            max_tokens=100
        )

        advice = response.choices[0].message.content
        print("\nNEMOTRON SAFETY ADVICE:")
        print(advice)

    except Exception as error:
        print("Nemotron API error:", error)

# -------- VIDEO SETUP --------
video = cv2.VideoCapture(VIDEO_FILE)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

if not video.isOpened():
    print("Error: driver.mp4 nahi mili!")
    raise SystemExit

fps = video.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 25

closed_start = None
alert_played = False
ai_requested = False

# -------- MAIN LOOP --------
while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    eyes_closed = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        face_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(
            face_gray, scaleFactor=1.1, minNeighbors=8, minSize=(15, 15)
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                frame,
                (x+ex, y+ey),
                (x+ex+ew, y+ey+eh),
                (255, 255, 0), 2
            )

        if len(eyes) == 0:
            eyes_closed = True

    if eyes_closed:
        if closed_start is None:
            closed_start = time.monotonic()

        elapsed = time.monotonic() - closed_start

        if elapsed >= ALERT_AFTER_SECONDS:
            cv2.putText(
                frame, "DROWSINESS ALERT!",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 3
            )

            if not alert_played:
                winsound.Beep(1800, 700)
                alert_played = True

            # Call Nemotron once per continuous alert
            if not ai_requested:
                threading.Thread(
                    target=ask_nemotron,
                    daemon=True
                ).start()
                ai_requested = True

    else:
        closed_start = None
        alert_played = False
        ai_requested = False

    cv2.imshow("Driver Drowsiness Detection + Nemotron", frame)

    if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

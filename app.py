"""
Driver Safety AI
Recorded video + live webcam detection.

Install:
    python -m pip install streamlit opencv-python-headless openai streamlit-webrtc streamlit-autorefresh imageio-ffmpeg

Run on Windows:
    set NEBIUS_API_KEY=YOUR_ACTUAL_API_KEY
    streamlit run app.py
"""

import os
import cv2
import time
import math
import hashlib
import threading
import tempfile
import subprocess
import wave
import struct
import io
from pathlib import Path

import av
import streamlit as st
from openai import OpenAI
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from streamlit_autorefresh import st_autorefresh


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Driver Safety AI",
    page_icon="🚗",
    layout="wide",
)

MODEL = "nvidia/nemotron-3-super-120b-a12b"
BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

FALLBACK_ADVICE = (
    "If you notice frequent or prolonged eye-closures while driving, "
    "pull over at a safe location and take a brief rest—stretch, "
    "hydrate, or nap until you feel fully alert before continuing."
)


# ============================================================
# API KEY AND AI ADVICE
# ============================================================

def get_api_key():
    """Read API key from environment variable or Streamlit secrets."""
    key = os.getenv("NEBIUS_API_KEY")

    if key:
        return key

    try:
        return st.secrets.get("NEBIUS_API_KEY")
    except Exception:
        return None


def get_ai_advice(
    alerts,
    longest,
    duration,
    source,
    previous_advice=None,
    alert_number=None,
):
    """Ask Nemotron for a recommendation, with a fallback if unavailable."""

    api_key = get_api_key()

    fallback_messages = [
        "Pull over at the next safe location, park, and take a rest before continuing.",
        "Do not try to push through tiredness. Stop somewhere safe, hydrate, and rest.",
        "When safe, park the vehicle and take a short nap or arrange another driver.",
        "Avoid continuing if you feel sleepy. Find a safe stopping place and rest first.",
        "Please pause the journey at a safe location, get some fresh air, and rest before driving again.",
    ]

    if not api_key:
        if alert_number is not None:
            return fallback_messages[
                (int(alert_number) - 1) % len(fallback_messages)
            ]
        return FALLBACK_ADVICE

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=BASE_URL,
            timeout=20.0,
            max_retries=1,
        )

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a driver safety assistant. Provide one short, "
                        "practical recommendation in plain English. Do not diagnose "
                        "medical conditions. If drowsiness is suspected, recommend "
                        "stopping at a safe location and resting. Eye detection "
                        "alone cannot prove that a driver is drowsy. Each alert "
                        "should receive fresh wording and a different practical "
                        "focus when possible."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Source: {source}. "
                        f"Possible alert episodes: {alerts}. "
                        f"Longest possible eye closure: {longest:.1f} seconds. "
                        f"Observed duration: {duration:.1f} seconds. "
                        f"Alert number: {alert_number if alert_number is not None else alerts}. "
                        f"Previous recommendation: {previous_advice or 'None provided'}. "
                        "Give a new, distinct, practical safety recommendation. "
                        "If drowsiness is suspected, prioritize stopping safely and resting."
                    ),
                },
            ],
            max_tokens=120,
            temperature=0.2,
        )

        advice = response.choices[0].message.content

        if advice and advice.strip():
            return advice.strip()

    except Exception:
        if alert_number is not None:
            return fallback_messages[
                (int(alert_number) - 1) % len(fallback_messages)
            ]

        return FALLBACK_ADVICE

    return FALLBACK_ADVICE


# ============================================================
# OPENCV FACE AND EYE DETECTION
# ============================================================

def load_detectors():
    """Load OpenCV face and eye Haar Cascade classifiers."""

    face_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )
    eye_path = (
        cv2.data.haarcascades
        + "haarcascade_eye.xml"
    )

    face_detector = cv2.CascadeClassifier(face_path)
    eye_detector = cv2.CascadeClassifier(eye_path)

    if face_detector.empty() or eye_detector.empty():
        raise RuntimeError(
            "OpenCV could not load its face/eye detector files. "
            "Reinstall OpenCV and try again."
        )

    return face_detector, eye_detector


def detect_frame(image, face_detector, eye_detector):
    """Detect faces and eyes; flag a possible eye-closure event."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    possible_closure = False

    for x, y, width, height in faces:
        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            (0, 200, 0),
            2,
        )

        upper_height = max(1, int(height * 0.65))
        eye_region = gray[
            y:y + upper_height,
            x:x + width,
        ]

        eyes = eye_detector.detectMultiScale(
            eye_region,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(15, 15),
        )

        for eye_x, eye_y, eye_width, eye_height in eyes:
            cv2.rectangle(
                image,
                (x + eye_x, y + eye_y),
                (
                    x + eye_x + eye_width,
                    y + eye_y + eye_height,
                ),
                (255, 200, 0),
                2,
            )

        if len(eyes) == 0:
            possible_closure = True

    return image, len(faces), possible_closure


# ============================================================
# LIVE WEBCAM METRICS
# ============================================================

class LiveMetrics:
    """Thread-safe metrics for webcam processing."""

    def __init__(self, alert_after=1.0):
        self.lock = threading.Lock()
        self.face_detector, self.eye_detector = load_detectors()

        self.alert_after = alert_after
        self.started = None
        self.last_frame_at = None
        self.frames = 0
        self.alerts = 0
        self.longest = 0.0
        self.closed_since = None
        self.alert_active = False
        self.face_seen = False
        self.events = []

    def set_alert_threshold(self, seconds):
        with self.lock:
            self.alert_after = seconds

    def reset(self):
        with self.lock:
            self.started = None
            self.last_frame_at = None
            self.frames = 0
            self.alerts = 0
            self.longest = 0.0
            self.closed_since = None
            self.alert_active = False
            self.face_seen = False
            self.events = []

    def process_frame(self, frame):
        image = frame.to_ndarray(format="bgr24")

        image, face_count, possible_closure = detect_frame(
            image,
            self.face_detector,
            self.eye_detector,
        )

        now = time.monotonic()

        with self.lock:
            if self.started is None:
                self.started = now

            self.last_frame_at = now
            self.frames += 1
            self.face_seen = face_count > 0

            if possible_closure:
                if self.closed_since is None:
                    self.closed_since = now

                closure_seconds = now - self.closed_since
                self.longest = max(self.longest, closure_seconds)

                if (
                    closure_seconds >= self.alert_after
                    and not self.alert_active
                ):
                    self.alerts += 1
                    self.alert_active = True

                    self.events.append({
                        "Alert #": self.alerts,
                        "Time in session (s)": round(
                            now - self.started, 2
                        ),
                        "Possible eye closure (s)": round(
                            closure_seconds, 2
                        ),
                        "Note": "Possible eye-closure event; not a diagnosis",
                    })

            else:
                self.closed_since = None
                self.alert_active = False

            alert_active = self.alert_active
            alert_count = self.alerts

        if alert_active:
            cv2.putText(
                image,
                "POSSIBLE DROWSINESS ALERT",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            image,
            f"Faces: {face_count} | Alerts: {alert_count}",
            (15, image.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )

    def snapshot(self):
        now = time.monotonic()

        with self.lock:
            frames = self.frames
            alerts = self.alerts
            longest = self.longest
            started = self.started
            last_frame_at = self.last_frame_at
            face_seen = self.face_seen
            events = [dict(event) for event in self.events]

        duration = 0.0

        if started is not None and last_frame_at is not None:
            if now - last_frame_at <= 2.0:
                duration = now - started
            else:
                duration = last_frame_at - started

        return {
            "frames": frames,
            "alerts": alerts,
            "longest": longest,
            "duration": max(0.0, duration),
            "face_seen": face_seen,
            "events": events,
        }


class LiveAdviceManager:
    """Generate webcam advice in a background thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active = True
        self.requested_alerts = 0
        self.completed_alerts = 0
        self.advice = None
        self.generating = False
        self.worker = None

    def reset(self):
        with self.lock:
            self.active = True
            self.requested_alerts = 0
            self.completed_alerts = 0
            self.advice = None
            self.generating = False
            self.worker = None

    def stop(self):
        with self.lock:
            self.active = False

    def request(self, alert_count, longest, duration):
        with self.lock:
            if not self.active:
                return

            self.requested_alerts = max(
                self.requested_alerts,
                int(alert_count),
            )

            if self.worker is not None and self.worker.is_alive():
                return

            if self.completed_alerts >= self.requested_alerts:
                return

            self.generating = True
            self.worker = threading.Thread(
                target=self._run,
                args=(float(longest), float(duration)),
                daemon=True,
            )
            self.worker.start()

    def _run(self, longest, duration):
        while True:
            with self.lock:
                if not self.active:
                    self.generating = False
                    return

                if self.completed_alerts >= self.requested_alerts:
                    self.generating = False
                    return

                alert_number = self.completed_alerts + 1
                previous_advice = self.advice

            advice = get_ai_advice(
                alert_number,
                longest,
                duration,
                "live webcam",
                previous_advice=previous_advice,
                alert_number=alert_number,
            )

            with self.lock:
                if not self.active:
                    self.generating = False
                    return

                self.advice = advice
                self.completed_alerts = alert_number

    def snapshot(self):
        with self.lock:
            return {
                "advice": self.advice,
                "completed_alerts": self.completed_alerts,
                "generating": self.generating,
            }


# ============================================================
# RECORDED VIDEO PROCESSING
# ============================================================

def process_video(input_path, raw_output_path, alert_after, progress):
    """Process a video using OpenCV and annotate possible eye closures."""

    face_detector, eye_detector = load_detectors()
    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise RuntimeError(
            "Could not open the uploaded video. Try another video file."
        )

    writer = None

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)

        if not fps or fps <= 0 or fps > 120:
            fps = 25.0

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            raise RuntimeError("The video dimensions are invalid.")

        writer = cv2.VideoWriter(
            str(raw_output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(
                "OpenCV could not create the intermediate output video."
            )

        frames_processed = 0
        alert_count = 0
        closed_frames = 0
        longest_closure = 0.0
        alert_active = False
        events = []

        while True:
            success, frame = capture.read()

            if not success:
                break

            frames_processed += 1

            frame, face_count, possible_closure = detect_frame(
                frame,
                face_detector,
                eye_detector,
            )

            if possible_closure:
                closed_frames += 1
                closure_seconds = closed_frames / fps

                longest_closure = max(
                    longest_closure,
                    closure_seconds,
                )

                if (
                    closure_seconds >= alert_after
                    and not alert_active
                ):
                    alert_count += 1
                    alert_active = True

                    events.append({
                        "Alert": alert_count,
                        "Frame": frames_processed,
                        "Video time (s)": round(
                            frames_processed / fps, 2
                        ),
                        "Possible closure (s)": round(
                            closure_seconds, 2
                        ),
                    })

            else:
                closed_frames = 0
                alert_active = False

            if alert_active:
                cv2.putText(
                    frame,
                    "POSSIBLE DROWSINESS ALERT",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                frame,
                (
                    f"Frame: {frames_processed} | "
                    f"Faces: {face_count} | Alerts: {alert_count}"
                ),
                (15, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

            writer.write(frame)

            if total_frames > 0 and frames_processed % 5 == 0:
                percent = min(
                    int(frames_processed * 100 / total_frames),
                    100,
                )
                progress.progress(
                    percent,
                    text=f"Processing video: {percent}%",
                )

        if frames_processed == 0:
            raise RuntimeError("No readable frames were found.")

        progress.progress(
            100,
            text="Detection complete. Preparing playable video...",
        )

        return {
            "frames": frames_processed,
            "fps": fps,
            "duration": frames_processed / fps,
            "alerts": alert_count,
            "longest": longest_closure,
            "events": events,
        }

    finally:
        capture.release()

        if writer is not None:
            writer.release()


# ============================================================
# CONVERT OUTPUT VIDEO TO H.264 MP4
# ============================================================

def convert_to_browser_mp4(raw_path, final_path):
    """Convert the intermediate video to browser-compatible H.264 MP4."""

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as error:
        raise RuntimeError(
            "FFmpeg was not found. Run: "
            "python -m pip install imageio-ffmpeg"
        ) from error

    for encoder in ("libx264", "libopenh264"):
        if encoder == "libx264":
            encoder_options = ["-preset", "veryfast", "-crf", "23"]
        else:
            encoder_options = ["-b:v", "2M"]

        command = [
            ffmpeg_exe,
            "-y",
            "-i", str(raw_path),
            "-map", "0:v:0",
            "-an",
            "-c:v", encoder,
            *encoder_options,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(final_path),
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=900,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "Video conversion timed out. Try a shorter or smaller video."
            ) from error

        if (
            completed.returncode == 0
            and final_path.exists()
            and final_path.stat().st_size > 0
        ):
            return

        if final_path.exists():
            final_path.unlink()

    raise RuntimeError(
        "FFmpeg could not encode H.264. Install a full FFmpeg build "
        "with a libx264 or libopenh264 encoder, then try again."
    )


# ============================================================
# ALERT SOUND
# ============================================================

def make_alert_beep():
    """Generate a short WAV alert sound."""

    sample_rate = 22050
    duration = 0.45
    frequency = 1500

    samples = b"".join(
        struct.pack(
            "<h",
            int(
                10000 * math.sin(
                    2 * math.pi * frequency * i / sample_rate
                )
            ),
        )
        for i in range(int(sample_rate * duration))
    )

    output = io.BytesIO()

    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples)

    return output.getvalue()


# ============================================================
# APP HEADER AND SIDEBAR
# ============================================================

st.title("🚗 Driver Safety AI")

st.caption(
    "Recorded video analysis and live webcam detection "
    "with AI safety advice."
)

with st.sidebar:
    st.header("Settings")

    alert_after = st.slider(
        "Alert after possible eye closure (seconds)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.5,
    )

    if get_api_key():
        st.success("Nebius API key found.")
    else:
        st.warning(
            "Nebius API key not found. General safety advice "
            "will still be available."
        )

    st.info(
        "This is a demonstration, not a certified safety device. "
        "Eye detection may produce false alerts."
    )

mode = st.radio(
    "Select mode",
    ["Upload / recorded video", "Live webcam"],
    horizontal=True,
)


# ============================================================
# LIVE WEBCAM
# ============================================================

if mode == "Live webcam":

    st.subheader("📷 Live Webcam Detection")

    st.write(
        "Click START and permit camera access in your browser. "
        "Test while parked, never while driving."
    )

    if "live_metrics" not in st.session_state:
        st.session_state["live_metrics"] = LiveMetrics(
            alert_after=alert_after
        )

    live_metrics = st.session_state["live_metrics"]
    live_metrics.set_alert_threshold(alert_after)

    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]

    try:
        turn_url = st.secrets.get("TURN_URL")
        turn_username = st.secrets.get("TURN_USERNAME")
        turn_password = st.secrets.get("TURN_PASSWORD")

        if turn_url and turn_username and turn_password:
            ice_servers.append({
                "urls": [turn_url],
                "username": turn_username,
                "credential": turn_password,
            })
    except Exception:
        pass

    webrtc_ctx = webrtc_streamer(
        key="driver-safety-live",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=live_metrics.process_frame,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        rtc_configuration={
            "iceServers": ice_servers,
        },
        async_processing=True,
    )

    camera_is_playing = bool(webrtc_ctx.state.playing)
    was_playing = st.session_state.get(
        "live_camera_was_playing",
        False,
    )

    if "live_advice_manager" not in st.session_state:
        st.session_state["live_advice_manager"] = LiveAdviceManager()

    live_advice_manager = st.session_state["live_advice_manager"]

    if camera_is_playing and not was_playing:
        live_metrics.reset()
        live_advice_manager = LiveAdviceManager()
        st.session_state["live_advice_manager"] = live_advice_manager
        st.session_state["live_notified_alert_count"] = 0

    elif not camera_is_playing and was_playing:
        live_advice_manager.stop()

    st.session_state["live_camera_was_playing"] = camera_is_playing

    if camera_is_playing:
        st_autorefresh(
            interval=1000,
            key="live-metrics-refresh",
        )
        st.success("Webcam session is running.")
    else:
        st.info(
            "Click START in the webcam panel. Advice is generated automatically "
            "for alerts while the camera is running."
        )

    live_snapshot = live_metrics.snapshot()

    st.subheader("Live Metrics")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Frames processed", f"{live_snapshot['frames']:,}")
    col2.metric("Possible alert episodes", live_snapshot["alerts"])
    col3.metric(
        "Longest possible closure",
        f"{live_snapshot['longest']:.1f} s",
    )
    col4.metric(
        "Observed session",
        f"{live_snapshot['duration']:.1f} s",
    )

    if "live_notified_alert_count" not in st.session_state:
        st.session_state["live_notified_alert_count"] = 0

    new_alert = (
        camera_is_playing
        and live_snapshot["alerts"]
        > st.session_state["live_notified_alert_count"]
    )

    if live_snapshot["frames"] == 0:
        st.caption(
            "Waiting for camera frames. If this remains at zero after START, "
            "check browser camera permission and webcam connection."
        )
    elif live_snapshot["alerts"] > 0:
        st.warning(
            "⚠️ Possible eye-closure alert detected. If you are driving, "
            "stop at a safe place before resting. This is not a medical diagnosis."
        )
    else:
        st.success(
            "No prolonged possible eye-closure alerts detected yet."
        )

    if new_alert:
        st.session_state["live_notified_alert_count"] = (
            live_snapshot["alerts"]
        )
        st.warning(
            "🔔 DRIVER SAFETY ALERT — possible drowsiness/eye closure."
        )
        st.audio(
            make_alert_beep(),
            format="audio/wav",
            autoplay=True,
        )

    if camera_is_playing and live_snapshot["alerts"] > 0:
        live_advice_manager.request(
            live_snapshot["alerts"],
            live_snapshot["longest"],
            live_snapshot["duration"],
        )

    advice_state = live_advice_manager.snapshot()

    if advice_state["generating"]:
        st.info(
            f"Alert advice is being prepared "
            f"({advice_state['completed_alerts']}/"
            f"{live_snapshot['alerts']} complete)…"
        )

    if advice_state["advice"]:
        st.subheader("AI Safety Advice")
        st.info(advice_state["advice"])

    if not camera_is_playing and live_snapshot["frames"] > 0:
        st.subheader("📋 Live Webcam Alert Log")

        if live_snapshot["events"]:
            st.dataframe(
                live_snapshot["events"],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success(
                "No possible eye-closure alert events were recorded."
            )

    st.caption(
        "Audio autoplay depends on browser permissions/settings. Visual warnings "
        "and the alert table remain available if the browser blocks sound. "
        "This demo is not a certified safety device."
    )


# ============================================================
# UPLOAD / RECORDED VIDEO — UPDATED PROCESSING FLOW
# ============================================================

else:

    uploaded_video = st.file_uploader(
        "Upload a driver video",
        type=["mp4", "avi", "mov", "mkv"],
        key="driver-video-upload",
    )

    video_bytes = (
        uploaded_video.getvalue()
        if uploaded_video is not None
        else None
    )

    video_filename = (
        uploaded_video.name
        if uploaded_video is not None
        else None
    )

    if "video_processing" not in st.session_state:
        st.session_state["video_processing"] = False

    if video_bytes is None:
        st.session_state["video_processing"] = False
        st.info(
            "📤 Please upload a video first. "
            "The Run button will appear after upload."
        )

    else:
        source_signature = hashlib.sha256(video_bytes).hexdigest()

        # Reset results only if the user selects a different video.
        if (
            st.session_state.get("source_signature")
            != source_signature
        ):
            st.session_state["source_signature"] = source_signature
            st.session_state.pop("processed_video", None)
            st.session_state.pop("video_result", None)
            st.session_state.pop("video_advice", None)
            st.session_state.pop("processed_signature", None)
            st.session_state["video_processing"] = False

        st.write(f"Selected video: **{video_filename}**")

        already_processed = (
            st.session_state.get("processed_signature")
            == source_signature
            and st.session_state.get("video_result") is not None
            and bool(st.session_state.get("processed_video"))
        )

        # ----------------------------------------------------
        # BUTTONS: only show them when processing is not active
        # ----------------------------------------------------

        if (
            not st.session_state["video_processing"]
            and not already_processed
        ):
            if st.button(
                "▶ Run Drowsiness Detection",
                type="primary",
                use_container_width=True,
                key="run-recorded-video",
            ):
                st.session_state["video_processing"] = True

                # This rerun starts processing on the next script run.
                st.rerun()

        elif (
            already_processed
            and not st.session_state["video_processing"]
        ):
            st.success(
                "✅ This video has already been processed. "
                "Results are shown below."
            )

            if st.button(
                "🔄 Process This Video Again",
                use_container_width=True,
                key="rerun-recorded-video",
            ):
                st.session_state.pop("processed_signature", None)
                st.session_state.pop("processed_video", None)
                st.session_state.pop("video_result", None)
                st.session_state.pop("video_advice", None)

                st.session_state["video_processing"] = True
                st.rerun()

        # ----------------------------------------------------
        # IMPORTANT FIX:
        # Process only when the processing flag is True.
        # This is NOT the button's "else" branch anymore.
        # ----------------------------------------------------

        if st.session_state["video_processing"]:

            progress = st.progress(
                0,
                text="Preparing video...",
            )

            st.info(
                "⏳ Video processing. Please wait..."
            )

            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir = Path(temp_dir)

                    suffix = Path(video_filename).suffix or ".mp4"

                    input_path = temp_dir / f"input{suffix}"
                    raw_output_path = temp_dir / "annotated_raw.mp4"
                    h264_output_path = (
                        temp_dir / "processed_driver_video.mp4"
                    )

                    input_path.write_bytes(video_bytes)

                    with st.spinner(
                        "Processing the complete video..."
                    ):
                        result = process_video(
                            input_path,
                            raw_output_path,
                            alert_after,
                            progress,
                        )

                    with st.spinner(
                        "Preparing playable video..."
                    ):
                        convert_to_browser_mp4(
                            raw_output_path,
                            h264_output_path,
                        )

                    processed_bytes = h264_output_path.read_bytes()

                if not processed_bytes:
                    raise RuntimeError(
                        "Processed video file is empty."
                    )

                # Save the completed result before displaying it.
                st.session_state["processed_video"] = processed_bytes
                st.session_state["video_result"] = result
                st.session_state["video_advice"] = None
                st.session_state["processed_signature"] = (
                    source_signature
                )

                st.success("✅ Video processing complete!")

            except Exception as error:
                st.error(
                    f"Video processing failed: "
                    f"{type(error).__name__}: {error}"
                )

                # Make the Run button available again after an error.
                st.session_state.pop("processed_video", None)
                st.session_state.pop("video_result", None)
                st.session_state.pop("video_advice", None)
                st.session_state.pop("processed_signature", None)

            finally:
                # Always reset the processing flag.
                st.session_state["video_processing"] = False

            # FIX: Do not call st.rerun() here.
            # Continue below and show the result immediately.

        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------

        if (
            video_bytes is not None
            and st.session_state.get("processed_signature")
            == source_signature
            and st.session_state.get("video_result") is not None
            and st.session_state.get("processed_video")
        ):
            result = st.session_state["video_result"]
            processed_bytes = st.session_state["processed_video"]

            st.subheader("Detection Summary")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Frames processed",
                f"{result['frames']:,}",
            )
            c2.metric(
                "Video duration",
                f"{result['duration']:.1f} s",
            )
            c3.metric(
                "Possible alerts",
                result["alerts"],
            )
            c4.metric(
                "Longest possible closure",
                f"{result['longest']:.1f} s",
            )

            video_column, advice_column = st.columns([1.4, 1])

            with video_column:
                st.subheader("Processed Video")

                st.video(
                    processed_bytes,
                    format="video/mp4",
                )

                st.download_button(
                    "⬇ Download processed video",
                    data=processed_bytes,
                    file_name="processed_driver_video.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    key="download-processed-video",
                )

            with advice_column:
                st.subheader("AI Safety Advice")

                if result["alerts"] > 0:
                    st.warning(
                        "Possible eye-closure alerts detected. "
                        "Take a safe rest break if drowsiness is suspected."
                    )

                    st.audio(
                        make_alert_beep(),
                        format="audio/wav",
                    )

                if st.button(
                    "🧠 Get AI Safety Advice",
                    type="primary",
                    use_container_width=True,
                    key="recorded-ai-advice-button",
                ):
                    with st.spinner(
                        "Preparing video safety advice..."
                    ):
                        st.session_state["video_advice"] = get_ai_advice(
                            result["alerts"],
                            result["longest"],
                            result["duration"],
                            "recorded video",
                        )

                if st.session_state.get("video_advice"):
                    st.info(
                        st.session_state["video_advice"]
                    )
                else:
                    st.caption(
                        "Click Get AI Safety Advice to display a recommendation."
                    )

            st.subheader("Alert Events")

            if result["events"]:
                st.dataframe(
                    result["events"],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success(
                    "No prolonged possible eye-closure alert was detected. "
                    "This does not guarantee that the driver was fully alert."
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Driver Safety AI is a demonstration only, not a certified safety "
    "system. Lighting, glasses, head position, and camera quality can "
    "cause false detections. Always prioritize safe driving decisions."
)
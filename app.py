import os, cv2, tempfile, io, wave, math, struct, time, threading
from pathlib import Path
import streamlit as st
from openai import OpenAI
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Driver Safety AI", page_icon="🚗", layout="wide")


MODEL = "nvidia/nemotron-3-super-120b-a12b"
BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

def make_alert_beep():
    rate, duration, freq = 22050, 0.45, 1500
    raw = b"".join(struct.pack("<h", int(10000*math.sin(2*math.pi*freq*i/rate))) for i in range(int(rate*duration))
    )
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(rate); wav.writeframes(raw)
    return out.getvalue()

def nemotron_advice(alerts, longest, duration, source="video"):
    key = os.getenv("NEBIUS_API_KEY")
    if not key:
        return "NEBIUS_API_KEY is not set. Detection still works without AI."
    try:
        client = OpenAI(api_key=key, base_url=BASE_URL, timeout=20)
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content":"You are a driver-safety demo assistant. Give one short practical recommendation. Do not diagnose medical conditions. If drowsiness is suspected, advise stopping safely to rest."},
                {"role":"user","content":f"Source: {source}; duration observed: {duration:.1f}s; possible eye-closure alert episodes: {alerts}; longest possible closure: {longest:.1f}s. Give one safety recommendation."}
            ], max_tokens=120)
        return r.choices[0].message.content or "No advice returned."
    except Exception as e:
        return f"Nemotron API error: {type(e).__name__}: {e}"

def cascades():
    face=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    eye=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")
    if face.empty() or eye.empty(): raise RuntimeError("OpenCV face/eye cascade files could not be loaded.")
    return face, eye

def detect_frame(image, face, eye):
    gray=cv2.equalizeHist(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
    faces=face.detectMultiScale(gray,1.1,5,minSize=(80,80))
    closed=False
    for x,y,w,h in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,200,0),2)
        top=max(1,int(h*.65))
        eyes=eye.detectMultiScale(gray[y:y+top,x:x+w],1.1,8,minSize=(15,15))
        for ex,ey,ew,eh in eyes:
            cv2.rectangle(image,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(255,200,0),2)
        if len(eyes)==0: closed=True
    return image, len(faces), closed

def process_video(src,dst,alert_after):
    face,eye=cascades(); cap=cv2.VideoCapture(str(src))
    if not cap.isOpened(): raise RuntimeError("Could not open video.")
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps<=0 or fps>120: fps=25.0
    w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if w<=0 or h<=0: cap.release(); raise RuntimeError("Invalid video dimensions.")
    writer=cv2.VideoWriter(str(dst),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    if not writer.isOpened(): cap.release(); raise RuntimeError("Could not create output video.")
    frames=alerts=closed_frames=0; longest=0.0; active=False; events=[]
    bar=st.progress(0,text="Processing video…")
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            frames+=1
            frame,nfaces,closed=detect_frame(frame,face,eye)
            if closed:
                closed_frames+=1; secs=closed_frames/fps; longest=max(longest,secs)
                if secs>=alert_after and not active:
                    alerts+=1; active=True
                    events.append({"Alert":alerts,"Frame":frames,"Video time (s)":round(frames/fps,2),"Possible closure (s)":round(secs,2)})
            else: closed_frames=0; active=False
            if active: cv2.putText(frame,"DROWSINESS ALERT!",(20,45),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,0,255),3)
            cv2.putText(frame,f"Frame: {frames} | Faces: {nfaces} | Alerts: {alerts}",(15,h-18),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
            writer.write(frame)
            if total>0 and frames%5==0:
                pct=min(int(frames*100/total),100); bar.progress(pct,text=f"Processing video… {pct}%")
    finally:
        cap.release(); writer.release()
    bar.progress(100,text="Processing complete.")
    return {"frames":frames,"fps":fps,"duration":frames/fps,"alerts":alerts,"longest":longest,"events":events}

class LiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.face,self.eye=cascades(); self.lock=threading.Lock()
        self.alert_after=1.0; self.closed_since=None; self.active=False
        self.alerts=0; self.longest=0.0; self.started=time.time(); self.frames=0; self.face_seen=False
    def recv(self,frame):
        image=frame.to_ndarray(format="bgr24")
        image,nfaces,closed=detect_frame(image,self.face,self.eye); now=time.time()
        with self.lock:
            self.frames+=1; self.face_seen=nfaces>0
            if closed:
                if self.closed_since is None: self.closed_since=now
                seconds=now-self.closed_since; self.longest=max(self.longest,seconds)
                if seconds>=self.alert_after and not self.active:
                    self.alerts+=1; self.active=True
            else: self.closed_since=None; self.active=False
            active=self.active; alerts=self.alerts
        if active: cv2.putText(image,"DROWSINESS ALERT!",(20,45),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,0,255),3)
        cv2.putText(image,f"LIVE | Faces: {nfaces} | Alerts: {alerts}",(15,image.shape[0]-18),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
        return type(frame).from_ndarray(image,format="bgr24")

st.title("🚗 Driver Safety AI")
st.caption("OpenCV Drowsiness Detection • Nebius Token Factory • NVIDIA Nemotron")
with st.sidebar:
    st.header("Settings")
    alert_after=st.slider("Alert after possible eye closure (seconds)",.5,3.0,1.0,.5)
    st.info("Demo only, not certified. Haar Cascade may miss partially closed eyes or produce false alerts.")
    # st.success("Nebius API key found.") if os.getenv("NEBIUS_API_KEY") else st.warning("API key not found. Detection still works without AI.")
    if os.getenv("NEBIUS_API_KEY"):
        st.success("Nebius API key found.")
    else:
        st.warning("API key not found. Detection still works without AI.")
mode=st.radio("Choose detection mode",["Upload / recorded video","Live webcam"],horizontal=True)
if mode=="Live webcam":
    st.subheader("📷 Live Webcam Drowsiness Detection")
    # st.write("Click START and allow camera access when your browser asks. Use only while parked, never while driving.")
    # ctx=webrtc_streamer(key="driver-safety-live",mode=WebRtcMode.SENDRECV,
    #     video_processor_factory=LiveProcessor,
    #     media_stream_constraints={"video":True,"audio":False},async_processing=True)
    
    ctx = webrtc_streamer(
        key="driver-safety-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LiveProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        # rtc_configuration={
        #     "iceServers": [
        #         {
        #             "urls": ["stun:stun.l.google.com:19302"]
        #         }
        #     ]
        # },

        rtc_configuration={
            "iceServers": [
                {
                    "urls": ["stun:stun.l.google.com:19302"]
                    },
                    {
                        "urls": [st.secrets["TURN_URL"]],
                        "username": st.secrets["TURN_USERNAME"],
                        "credential": st.secrets["TURN_PASSWORD"],
                    },
                    ]
                },
        async_processing=True
    )

    if ctx.video_processor:
        p=ctx.video_processor; p.alert_after=alert_after
        st_autorefresh(interval=1000,key="live-refresh")
        with p.lock:
            alerts=p.alerts; longest=p.longest; frames=p.frames; elapsed=time.time()-p.started; face_seen=p.face_seen
        st.subheader("Live Metrics")
        a,b,c,d=st.columns(4); a.metric("Frames processed",f"{frames:,}"); b.metric("Possible alert episodes",alerts)
        c.metric("Longest possible closure",f"{longest:.1f} s"); d.metric("Session time",f"{elapsed:.0f} s")
        # st.warning("Possible drowsiness detected. If this is a real driving situation, pull over safely before resting.") if alerts else st.success("No prolonged possible eye-closure alert yet.")
        
        if alerts:
            st.warning(
            "Possible drowsiness detected. If this is a real driving situation, "
            "pull over safely before resting."
            )
        else:
            st.success("No prolonged possible eye-closure alert yet.")

        # if st.button("🧠 Get AI safety advice for this live session"):
        #     with st.spinner("Contacting Nebius Token Factory…"):
        #         st.session_state["live_advice"]=nemotron_advice(alerts,longest,elapsed,"live webcam")
        # if "live_advice" in st.session_state: st.info(st.session_state["live_advice"])
        st.caption("No video is saved by this live mode. Detection can be affected by lighting, glasses, head turns, and camera position.")
    else: st.info("Click START in the camera panel to begin live detection.")
else:
    uploaded=st.file_uploader("Upload a driver video",type=["mp4","avi","mov","mkv"])
    default_video=Path(__file__).resolve().parent/"driver.mp4"
    if uploaded is not None: data,filename=uploaded.getvalue(),uploaded.name
    elif default_video.is_file():
        data,filename=default_video.read_bytes(),default_video.name
        st.caption("Using driver.mp4 from the app folder.")
    else: data,filename=None,None; st.info("Upload a video, or put driver.mp4 beside app.py.")
    if data is not None:
        st.write(f"Selected video: **{filename}**")
        if st.button("▶ Run Drowsiness Detection",type="primary",use_container_width=True):
            with tempfile.TemporaryDirectory() as td:
                td=Path(td); src=td/("input"+(Path(filename).suffix or ".mp4")); dst=td/"processed.mp4"; src.write_bytes(data)
                try:
                    with st.spinner("OpenCV is processing the video…"): result=process_video(src,dst,alert_after)
                    st.session_state["processed_video"]=dst.read_bytes(); st.session_state["result"]=result
                    st.session_state.pop("ai_advice",None)
                except Exception as e: st.error(f"Video processing error: {type(e).__name__}: {e}"); st.stop()
    if "result" in st.session_state:
        r=st.session_state["result"]; st.subheader("Detection Summary")
        if r["alerts"]>0:
            st.warning("Possible drowsiness detected. Take a safe rest break if this reflects a real driving situation.")
            st.audio(make_alert_beep(),format="audio/wav",autoplay=True)
        a,b,c,d=st.columns(4); a.metric("Frames processed",f"{r['frames']:,}"); b.metric("Video duration",f"{r['duration']:.1f} s")
        c.metric("Possible alert episodes",r["alerts"]); d.metric("Longest possible closure",f"{r['longest']:.1f} s")
        left,right=st.columns([1.4,1])
        with left:
            st.subheader("Processed video"); st.video(st.session_state["processed_video"])
            st.download_button("⬇ Download processed video",st.session_state["processed_video"],"processed_driver_video.mp4","video/mp4",use_container_width=True)
        with right:
            st.subheader("Nemotron Safety Advice")
            if st.button("🧠 Get AI safety advice",use_container_width=True):
                with st.spinner("Contacting Nebius Token Factory…"):
                    st.session_state["ai_advice"]=nemotron_advice(r["alerts"],r["longest"],r["duration"],"recorded video")
            if "ai_advice" in st.session_state: st.info(st.session_state["ai_advice"])
            else: st.caption("Run AI advice after video processing.")
        st.subheader("Alert events")
        if r["events"]: st.dataframe(r["events"],use_container_width=True,hide_index=True)
        else: st.success("No possible prolonged eye-closure alerts detected."); st.caption("This does not guarantee the driver was fully alert.")

st.divider()
st.caption("Eye-detection failure can result from lighting, glasses, head turns, or partial closure. This demo is not certified and must never be the only protection against drowsy driving.")

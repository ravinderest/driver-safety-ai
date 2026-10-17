# Driver Safety AI 🚘

**An AI-assisted driver-monitoring system designed to identify possible signs of driver drowsiness and provide safety alerts using OpenCV, Streamlit, and NVIDIA Nemotron.**

## 🔗 Live Demo
**Try the deployed application:**
👉 https://driver-safety-ai-5cg2a7ebssx2etsejup3hi.streamlit.app/

## Overview

Driver Safety AI is an AI-assisted driver-monitoring prototype that helps identify possible signs of drowsiness by monitoring a driver's face and eyes through a live webcam or recorded video.

When the system detects possible prolonged eye closure or repeated eye-detection failures, it can flag a potential drowsiness event and display alert information on the dashboard. This can help bring attention to situations where a driver may be feeling sleepy.

The application also provides monitoring metrics and can generate practical driver-safety recommendations using NVIDIA Nemotron.

**Goal:** Help improve driver awareness and road safety by highlighting possible signs of drowsiness before they become dangerous.

## Key Features

- **Possible Drowsiness Detection:** Uses face and eye detection to flag possible signs of driver sleepiness.
- **Driver Safety Alerts:** Displays possible eye-closure events and related alert information.
- **Live Webcam Monitoring:** Monitors a live webcam stream using WebRTC.
- **Recorded Video Analysis:** Processes uploaded or sample driving videos.
- **Detection Metrics:** Shows alert counts, session duration, and possible eye-closure information.
- **Annotated Video Output:** Provides processed video output where supported by the recorded-video workflow.
- **AI-Powered Safety Advice:** Can request practical safety recommendations from NVIDIA Nemotron.
- **Interactive Dashboard:** Uses Streamlit to present monitoring information in an accessible interface.

## How It Works

1. **Video Input:** The user starts live webcam monitoring or provides a recorded video.
2. **Face and Eye Detection:** OpenCV Haar Cascade classifiers process video frames to detect faces and eyes.
3. **Possible Drowsiness Indicators:** The application uses missing eye detections as a possible indication of eye closure.
4. **Alert and Metrics:** The dashboard presents possible alert events and session statistics.
5. **AI Safety Recommendation:** When requested, NVIDIA Nemotron can generate a practical safety recommendation based on the available session information.

## Technology Stack

- **Python:** Application logic
- **OpenCV:** Face and eye detection
- **Streamlit:** Web dashboard
- **streamlit-webrtc:** Live webcam streaming
- **Nebius Token Factory:** API access to the configured language model
- **NVIDIA Nemotron:** AI-generated driver-safety recommendations

## Installation

### 1. Install Dependencies

Make sure Python is installed, then open a terminal in the project folder and run:

```bash
python -m pip install -r requirements.txt
```

### 2. Configure the API Key

The NVIDIA Nemotron integration uses the Nebius Token Factory API.

In Windows PowerShell, set your API key with:

```powershell
$env:NEBIUS_API_KEY="YOUR_API_KEY"
```

Replace `YOUR_API_KEY` with your actual key. Keep it private and never commit it to GitHub.

The core detection functionality can operate without the AI recommendation feature, depending on the application configuration.

### 3. Run the Application

Start the Streamlit dashboard:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal. Allow webcam access when prompted if you want to use live monitoring.

## NVIDIA Nemotron Integration

The application uses the OpenAI Python SDK with the Nebius Token Factory API endpoint:

`https://api.tokenfactory.nebius.com/v1/`

Configured model:

`nvidia/nemotron-3-super-120b-a12b`

The application can send available session-level information, such as possible alert counts, longest observed possible eye closure, session duration, and input mode, to the model.

Nemotron generates a short, practical driver-safety recommendation. OpenCV remains responsible for the underlying video-frame detection; the language model provides additional guidance rather than performing the eye detection itself.

The API key must be stored in an environment variable and must never be committed to the repository.

## Benefits

- Helps draw attention to possible signs of driver sleepiness.
- Provides visible monitoring information during a demonstration.
- Combines computer vision with AI-generated safety guidance.
- Demonstrates how AI technologies can support driver-awareness applications.
- Provides a foundation for future improvements in driver-monitoring systems.

## Limitations and Safety Disclaimer

This project is a demonstration prototype, not a certified driver-monitoring or drowsiness-detection system.

The current method relies on Haar Cascade face and eye detection. Missing eye detections do not necessarily mean that a driver is asleep. Head movement, glasses, poor lighting, camera positioning, or occlusion can cause false alerts or missed events.

The system does not reliably establish whether a person is sleepy, asleep, or safe to continue driving. It must not be relied upon as the sole means of preventing accidents.

**Safety first:** Drivers should stop in a safe location and rest whenever they feel sleepy. Do not continue driving based only on the absence of an alert from this prototype.

## Future Improvements

- More reliable eye-state estimation using facial landmarks or trained models.
- Improved drowsiness estimation using eye-closure duration and temporal patterns.
- Configurable visual and audio alerts.
- Testing across different lighting conditions, camera angles, and eyewear.
- Better alert-event visualization and reporting.
- More robust error handling and deployment testing.
- Evaluation using labelled data to measure false positives and missed detections.

## Demo Video

The project demonstration should show:

1. The main dashboard.
2. Live webcam monitoring.
3. Recorded-video processing.
4. Possible eye-closure alerts and monitoring metrics.
5. NVIDIA Nemotron safety recommendations.

Only include features and performance results that have actually been tested and verified.

## Project Status

This is an experimental prototype intended for learning, demonstration, and further development in AI-assisted driver safety.


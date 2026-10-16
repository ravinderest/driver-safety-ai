# Driver Safety AI

**An AI-assisted driver-monitoring prototype using OpenCV, Streamlit, and NVIDIA Nemotron.**

## Overview

Driver Safety AI demonstrates how computer vision can flag possible prolonged eye-closure events from a recorded video or live webcam stream. It provides a dashboard for monitoring detection metrics and can request a practical driver-safety recommendation from NVIDIA Nemotron.

## Key Features

- Live webcam monitoring through WebRTC.
- Recorded-video processing and annotated output.
- OpenCV Haar Cascade face and eye detection.
- Possible eye-closure alert counters and session metrics.
- Optional AI-generated safety recommendations.
- Streamlit dashboard for presenting results.

## Technology Stack

- **Python:** application logic
- **OpenCV:** face and eye-region detection
- **Streamlit:** web dashboard
- **streamlit-webrtc:** live webcam streaming
- **Nebius Token Factory:** API access
- **NVIDIA Nemotron:** safety-advice generation

## Installation

Install Python and the project dependencies:

```bash
python -m pip install -r requirements_live.txt
```

Set the API key as an environment variable.

Windows PowerShell:

```powershell
$env:NEBIUS_API_KEY="YOUR_API_KEY"
```

Start the application:

```bash
python -m streamlit run app_live.py
```

Open the local URL printed in the terminal and allow webcam access when prompted.

## NVIDIA Nemotron Integration

The application uses the OpenAI Python SDK with the Nebius Token Factory API endpoint. It sends session-level information, such as the number of possible alert episodes, longest observed possible eye closure, session duration, and input mode.

The configured model is `nvidia/nemotron-3-super-120b-a12b`. Nemotron returns a short practical safety recommendation. The model provides advice; OpenCV performs the actual frame-level detection.

The API key must be stored in an environment variable and must never be committed to the repository.

## Limitations

The current detection method uses Haar Cascade classifiers and treats missing eye detections as possible eye closure. Head movement, glasses, lighting, occlusion, and detector failure can cause false alerts or missed events.

This is a demonstration prototype, not a certified driver-monitoring system, medical diagnostic tool, or substitute for safe driving practices.

## Future Improvements

- Landmark-based eye-state estimation.
- Systematic evaluation across lighting and camera conditions.
- Better alert-event visualization and reporting.
- Deployment testing and improved error handling.

## Demo

A short demonstration should show the dashboard, live webcam mode, recorded-video processing, alert metrics, and the optional Nemotron safety recommendation.

Only report test results and performance measurements that have actually been verified.
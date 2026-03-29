# 🧠 NeuroPose

> Real-time cognitive attention monitoring through facial behaviour analysis

![Python](https://img.shields.io/badge/Python-3.11-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-orange)

## What is NeuroPose?

NeuroPose is a real-time attention monitoring system that uses your
webcam to track study focus — no wearable hardware needed. It uses
MediaPipe's 468-point face mesh to analyse facial behaviour and compute
a live Focus Score every frame.

## Features

- Real-time face mesh with 468 3D landmarks at 30 fps
- Eye Aspect Ratio (EAR) for blink detection
- Iris-based gaze estimation (horizontal and vertical)
- 6-DoF head pose via OpenCV PnP solver and Rodrigues decomposition
- Mouth Aspect Ratio (MAR) for yawn detection
- Multi-signal weighted Focus Score with EMA smoothing
- Live Streamlit dashboard with Plotly gauge and timeline
- SQLite session logging with full event history
- Automated PDF session reports with embedded charts
- Pomodoro timer integration
- Pytest unit test suite

## Tech Stack

| Tool           | Purpose                              |
| -------------- | ------------------------------------ |
| Python 3.11    | Core language                        |
| MediaPipe 0.10 | Face mesh and iris landmarks         |
| OpenCV 4.9     | Camera, PnP solver, image processing |
| Streamlit 1.35 | Web dashboard                        |
| Plotly         | Live charts and gauge                |
| SQLite         | Session database                     |
| ReportLab      | PDF report generation                |
| Pytest         | Unit testing                         |

## How the Focus Score Works

Five signals are combined with weighted averaging:

| Signal             | Weight | What it measures                     |
| ------------------ | ------ | ------------------------------------ |
| Gaze direction     | 35%    | Is the person looking at the screen? |
| Head pose          | 25%    | Is the head oriented to the screen?  |
| Eye openness (EAR) | 20%    | Are eyes open?                       |
| Blink rate         | 12%    | 12-20 blinks/min = healthy           |
| Yawn (MAR)         | 8%     | Mouth open = fatigue                 |

The raw score is smoothed using Exponential Moving Average (alpha=0.08)
to prevent jitter.



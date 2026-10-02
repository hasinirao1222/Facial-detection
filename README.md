# Real-Time Facial Detection System

A simple real-time face detection project built with Python and OpenCV.

## Features

- Detects faces from a live webcam stream
- Draws bounding boxes around detected faces
- Uses an improved OpenCV pipeline with frontal and profile face cascades
- Applies contrast enhancement plus merged detections for better robustness
- Uses configurable thresholds for smoother real-time performance
- Displays live face count and FPS
- Accepts a webcam index or a video file as input
- Saves an annotated detection snapshot when you press `S`

## Project Structure

```text
face_detection_system/
|-- requirements.txt
|-- README.md
`-- src/
    |-- app.py
    `-- detector.py
```

## Setup

1. Create a virtual environment:

```powershell
python -m venv .venv
```

2. Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python src\app.py
```

Use a different camera index or process a video file with `--source`:

```powershell
python src\app.py --source 1
python src\app.py --source "videos\sample.mp4"
```

Press `S` to save the current annotated frame under `snapshots/` (or choose a
different directory with `--snapshot-dir`). Press `Q` to close the window.

## Resume Description

- Built a real-time face detection system using Python and OpenCV for accurate bounding-box detection.
- Improved detection accuracy by combining frontal and profile OpenCV cascades with merged detections.
- Optimized live video processing with configurable frame scaling for smoother inference.
- Improved robustness in varying lighting conditions and face positions using confidence-based detection.

## Future Improvements

- Replace Haar cascades with a DNN-based detector for better accuracy
- Add face recognition on top of detection
- Save snapshots when faces are detected

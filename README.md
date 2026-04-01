# Real-Time Facial Detection System

A simple real-time face detection project built with Python and OpenCV.

## Features

- Detects faces from a live webcam stream
- Draws bounding boxes around detected faces
- Uses an improved OpenCV pipeline with frontal and profile face cascades
- Applies contrast enhancement plus merged detections for better robustness
- Uses configurable thresholds for smoother real-time performance
- Displays live face count and FPS

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

Press `Q` to close the webcam window.

## Resume Description

- Built a real-time face detection system using Python and OpenCV for accurate bounding-box detection.
- Improved detection accuracy by combining frontal and profile OpenCV cascades with merged detections.
- Optimized live video processing with configurable frame scaling for smoother inference.
- Improved robustness in varying lighting conditions and face positions using confidence-based detection.

## Future Improvements

- Replace Haar cascades with a DNN-based detector for better accuracy
- Add face recognition on top of detection
- Save snapshots when faces are detected
- Support video-file input in addition to webcam input

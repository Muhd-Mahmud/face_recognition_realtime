# Face Recognition RealTime

A real-time face recognition system with Firebase database integration.

## Features
- Real-time face detection and recognition via webcam
- Firebase Realtime Database integration
- Face encoding generation and storage

## Files
- `main.py` — Main script for real-time face recognition
- `EncodeGenerator.py` — Generates and stores face encodings

## Requirements
```bash
pip install face_recognition opencv-python firebase-admin
```

## Usage
1. Add face images to the `Images/` folder
2. Run `EncodeGenerator.py` to generate encodings
3. Run `main.py` to start real-time recognition

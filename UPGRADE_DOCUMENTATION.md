# AI Virtual Mouse - Upgrade and Migration Documentation

This document records the analysis, architectural changes, library compatibility fixes, and usage instructions for the **AI Virtual Mouse Using Hand Gestures** project to run smoothly on the latest installed libraries (**MediaPipe 1.0.1+**, **OpenCV 5.0.0**, **NumPy 2.5.3**, **AutoPy 4.0.2**, **PyAutoGUI 0.9.54**, and **Python 3.14**).

---

## 1. Problem Summary & Root Causes

When attempting to run the original project files on modern Python environments and current library releases, the following critical blockers were identified:

1. **MediaPipe 1.0+ Breaking API Removal**:
   - The original code depended on `mediapipe.solutions.hands` and `mediapipe.solutions.drawing_utils`.
   - In MediaPipe 1.0.0+, the legacy `solutions` module was **completely removed** in favor of the **MediaPipe Tasks Vision API** (`mediapipe.tasks.python.vision.HandLandmarker`).
   - Running the original code triggered:
     ```text
     AttributeError: module 'mediapipe' has no attribute 'solutions'
     ```
2. **Missing MediaPipe Model Asset**:
   - The modern Tasks API requires a compiled `.task` model (`hand_landmarker.task`) to perform landmark detection.
3. **Missing `requirements.txt`**:
   - While referenced in `README.md`, `requirements.txt` was not included in the repository.
4. **AutoPy Boundary Clamping**:
   - Calls to `autopy.mouse.move(x, y)` threw `ValueError: Point out of bounds` whenever cursor coordinates slightly exceeded the screen boundaries `[0, wScr - 1]` or `[0, hScr - 1]`.
5. **Frame Drop and Drive Access Crashes**:
   - If the webcam dropped a frame, the script attempted to process `None`, resulting in crashes.
   - Calling `os.startfile("A:\\")` crashed systems lacking drive `A:\`.

---

## 2. Summary of Changes Implemented

### A. Core Engine: `HandTrackingModule.py`
- **Migrated to MediaPipe Tasks API**:
  - Replaced legacy `mp.solutions.hands.Hands` with `mediapipe.tasks.python.vision.HandLandmarker.create_from_options(...)`.
  - Configured `RunningMode.IMAGE` with support for configurable `num_hands`, `min_hand_detection_confidence`, and `min_tracking_confidence`.
- **Self-Healing Model Downloader (`ensure_model_file`)**:
  - Automatically verifies if `hand_landmarker.task` is present locally.
  - If missing, automatically downloads Google's official model from `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`.
- **Native OpenCV Rendering**:
  - Implemented direct OpenCV line and landmark rendering across all 21 keypoints and joints. This eliminates external drawing dependencies and delivers high FPS performance.
- **Defensive API Guards**:
  - Guarded `fingersUp()` against empty landmark lists to avoid `IndexError`.
  - Maintained 100% backward compatibility for methods: `findHands()`, `findPosition()`, `fingersUp()`, and `findDistance()`.

### B. Main Application: `aivirtualmouseproject.py`
- **Coordinate Boundary Clamping**:
  - Added `np.clip(..., 0, wScr - 1)` and `np.clip(..., 0, hScr - 1)` before calling `autopy.mouse.move()`, preventing out-of-bounds crashes when fingers move near camera borders.
- **Camera Frame Validation**:
  - Added `if not success or img is None: continue` to gracefully handle dropped camera frames.
- **Safe OS Action**:
  - Protected `os.startfile("A:\\")` inside a `try...except` block.
- **Integer Screen Coordinates**:
  - Converted screen dimensions from `autopy.screen.size()` to standard integers.

### C. Volume Controller: `volume.py`
- **Decoupled Mouse Movement**:
  - Removed unwanted mouse cursor movement that previously hijacked the cursor whenever the user pinched or spread their fingers to adjust volume.
  - Dedicates `volume.py` purely to gesture volume control, preventing mouse jumping.
- **On-Screen Volume HUD Bar**:
  - Added a live visual volume bar (0% - 100%) and percentage display directly on the camera preview window for immediate real-time feedback.
- **Smoother Stepping**:
  - Balanced volume up/down key events so system volume changes gracefully without lag or stutter.

### D. Dependency Manifest: `requirements.txt`
Created the missing dependency file with all five required libraries:
```text
opencv-python
mediapipe
numpy
autopy
pyautogui
```

---

## 3. Gesture Controls Reference

| Gesture | Finger State | Action |
| :--- | :--- | :--- |
| **Move Cursor** | Only Index finger extended (`fingers[1] == 1, fingers[2] == 0`) | Moves mouse smoothly across the screen |
| **Left Click** | Index & Middle fingers extended + pinched together (< 40 px) | Performs a left mouse click |
| **Right Click** | Index & Middle fingers extended + Thumb & Pinky tips pinched (< 40 px) | Performs a right mouse click |
| **Open Drive** | Index & Middle fingers extended + Thumb & Ring tips pinched (< 40 px) | Attempts to open drive `A:\` (fails safely if absent) |
| **Volume Control** *(in `volume.py`)* | Index & Middle fingers extended + vary distance between Thumb and Index | Increases or decreases system volume |
| **Exit** | Press `Esc` key on keyboard | Closes camera window and terminates program |

---

## 4. How to Run

### 1. Main Virtual Mouse Controller
```bash
python aivirtualmouseproject.py
```

### 2. Gesture Volume Controller
```bash
python volume.py
```

### 3. Hand Tracking Module Standalone Diagnostic
```bash
python HandTrackingModule.py
```

---

## 5. Verification & Test Results
- **Module Import & Initialization**: Verified via Python 3.14 — `HandLandmarker` delegate initializes successfully on CPU with XNNPACK.
- **Synthetic Inference**: Validated detection and drawing on blank/synthetic frames without errors.
- **Model Download**: `hand_landmarker.task` (7.8 MB) is verified and cached locally in the project root.

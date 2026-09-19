# AI Adaptive Virtual Mouse

An intelligent, contactless computer interaction system that translates natural hand gestures into real-time mouse movements, clicks, shortcuts, and volume adjustments using computer vision and deep learning.

Built with **MediaPipe Tasks Vision API (1.0+)**, **OpenCV**, **NumPy**, **AutoPy**, and **PyAutoGUI**.

---

## ✨ Features

- **Smooth Cursor Navigation**: Real-time cursor motion with mathematical interpolation and adaptive smoothing.
- **Natural Gesture Clicks**:
  - **Left Click**: Pinch Index and Middle fingers together.
  - **Right Click**: Pinch Thumb and Pinky fingers together.
  - **Custom Shortcuts**: Dedicated gestures for rapid system actions.
- **Dedicated Gesture Volume Control**:
  - Adjust system volume naturally by varying the distance between Thumb and Index fingers.
  - Visual on-screen HUD with real-time level gauge and volume percentage.
  - Decoupled cursor control to prevent unwanted mouse jumping during audio adjustment.
- **Modern MediaPipe Tasks Engine**:
  - Powered by the latest MediaPipe Vision Tasks framework (`HandLandmarker`).
  - Automatic model downloading (`hand_landmarker.task`) directly from Google's official repository on first execution.
- **Screen Boundary Safety**: Clamps coordinates to display dimensions to prevent out-of-bounds exceptions.

---

## 📁 Repository Structure

```text
├── hand_tracker.py             # Core MediaPipe Tasks hand tracking engine
├── adaptive_virtual_mouse.py   # Main virtual mouse & gesture click application
├── gesture_volume.py           # Standalone gesture volume controller with on-screen HUD
├── requirements.txt            # Dependency manifest
├── UPGRADE_DOCUMENTATION.md    # In-depth technical upgrade and architecture documentation
├── .gitignore                  # Git ignore rules for cached models & bytecode
└── README.md                   # Project overview and documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/TheMalyala/AI-Virtual-Mouse.git
cd AI-Virtual-Mouse
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed (tested through Python 3.14):
```bash
pip install -r requirements.txt
```

### 3. Run Applications

#### AI Virtual Mouse
```bash
python adaptive_virtual_mouse.py
```

#### Gesture Volume Controller
```bash
python gesture_volume.py
```

#### Hand Tracker Standalone Diagnostic
```bash
python hand_tracker.py
```

*(Press **`Esc`** at any time while the camera window is active to exit).*

---

## ✋ Gesture Controls Reference

### Virtual Mouse (`adaptive_virtual_mouse.py`)

| Action | Finger Gesture | Visual Cue |
| :--- | :--- | :--- |
| **Move Pointer** | Only **Index finger** extended (Middle finger down) | Purple cursor circle on Index tip |
| **Left Click** | Both **Index & Middle fingers** extended + pinched together (< 40 px) | Green indicator circle |
| **Right Click** | Both **Index & Middle fingers** extended + **Thumb & Pinky** pinched (< 40 px) | Green indicator circle |
| **Open Drive** | Both **Index & Middle fingers** extended + **Thumb & Ring** pinched (< 40 px) | Green indicator circle |

### Gesture Volume Controller (`gesture_volume.py`)

| Action | Finger Gesture | Visual Cue |
| :--- | :--- | :--- |
| **Increase Volume** | Extend **Index finger** and spread **Thumb** outward | Green volume gauge rises |
| **Decrease Volume** | Extend **Index finger** and pinch **Thumb** closer | Green volume gauge lowers |
| **Mute / Min** | Pinch **Thumb and Index** within 25 px | Green filled circle indicator |

---

## 🛠️ System Requirements

- **Operating System**: Windows 10/11
- **Hardware**: Integrated webcam or external USB camera
- **Python**: 3.8 to 3.14+

---

## 👤 Author

Created & maintained by **[TheMalyala](https://github.com/TheMalyala)**.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

"""
hand_tracker.py
High-performance hand landmark tracking engine built on MediaPipe Tasks Vision API (1.0+).
Provides real-time landmark detection, joint connections, finger elevation calculation,
and landmark distance measurement.
"""

import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math
import numpy as np

# Standard hand landmark connections across 21 joints
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def ensure_model_file(model_path=DEFAULT_MODEL_PATH):
    """Ensures that the hand landmarker model task file exists, downloading it if missing."""
    if not os.path.exists(model_path):
        print(f"Downloading hand landmarker model to {model_path} ...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("Model download complete.")
    return model_path


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5, model_path=DEFAULT_MODEL_PATH):
        self.mode = mode
        self.max_hands = int(max_hands)
        self.detection_con = float(detection_con)
        self.track_con = float(track_con)
        self.model_path = ensure_model_file(model_path)

        # Initialize MediaPipe Tasks HandLandmarker
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.tip_ids = [4, 8, 12, 16, 20]
        self.results = None
        self.lm_list = []

    def find_hands(self, img, draw=True):
        """Processes the input image to find hand landmarks and optionally draws them."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        self.results = self.detector.detect(mp_image)

        if self.results and self.results.hand_landmarks:
            h, w, _ = img.shape
            for hand_lms in self.results.hand_landmarks:
                if draw:
                    # Draw connection lines between landmarks
                    for start_idx, end_idx in HAND_CONNECTIONS:
                        pt1 = (int(hand_lms[start_idx].x * w), int(hand_lms[start_idx].y * h))
                        pt2 = (int(hand_lms[end_idx].x * w), int(hand_lms[end_idx].y * h))
                        cv2.line(img, pt1, pt2, (0, 255, 0), 2)
                    # Draw individual landmark circles
                    for lm in hand_lms:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return img

    def find_position(self, img, hand_no=0, draw=True):
        """Extracts 21 landmark positions and bounding box for a specific hand."""
        x_list = []
        y_list = []
        bbox = []
        self.lm_list = []

        if self.results and self.results.hand_landmarks:
            if hand_no < len(self.results.hand_landmarks):
                my_hand = self.results.hand_landmarks[hand_no]
                h, w, _ = img.shape
                for id, lm in enumerate(my_hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    x_list.append(cx)
                    y_list.append(cy)
                    self.lm_list.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                if x_list and y_list:
                    xmin, xmax = min(x_list), max(x_list)
                    ymin, ymax = min(y_list), max(y_list)
                    bbox = xmin, ymin, xmax, ymax

                    if draw:
                        cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                                      (0, 255, 0), 2)

        return self.lm_list, bbox

    def fingers_up(self):
        """Returns a list of 5 elements (1 for extended, 0 for curled) for Thumb, Index, Middle, Ring, Pinky."""
        fingers = []
        if not self.lm_list or len(self.lm_list) < 21:
            return [0, 0, 0, 0, 0]

        # Thumb: compare x coordinates
        if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers: compare y coordinates with PIP joints
        for id in range(1, 5):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def find_distance(self, p1, p2, img, draw=True, r=15, t=3):
        """Calculates Euclidean distance between two landmarks p1 and p2."""
        if len(self.lm_list) <= max(p1, p2):
            return 0, img, [0, 0, 0, 0, 0, 0]

        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

    # Aliases for backward compatibility
    findHands = find_hands
    findPosition = find_position
    fingersUp = fingers_up
    findDistance = find_distance


# Class alias for backward compatibility
handDetector = HandDetector


def main():
    p_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        if not success:
            break
        img = detector.find_hands(img)
        lm_list, bbox = detector.find_position(img)
        if len(lm_list) != 0:
            print("Thumb tip position:", lm_list[4])

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Hand Tracker Preview", img)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

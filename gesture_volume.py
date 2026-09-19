"""
gesture_volume.py
Vision-based Gesture Audio Volume Controller using MediaPipe Tasks and PyAutoGUI.
Provides on-screen visual HUD and decoupled cursor control for smooth audio adjustments.
Author: TheMalyala
"""

import time
import cv2
import numpy as np
import pyautogui
import hand_tracker as ht


def main():
    wCam, hCam = 640, 480

    pTime = 0
    previous_vol = 0
    volBar = 400
    volPer = 0

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    detector = ht.HandDetector(max_hands=1)

    print("AI Gesture Volume Controller started. Press ESC in the camera window to exit.")

    while True:
        # 1. Capture camera frame
        success, img = cap.read()
        if not success or img is None:
            continue

        # 2. Detect hand landmarks
        img = detector.find_hands(img)
        lm_list, bbox = detector.find_position(img, draw=False)

        # 3. Process volume gestures
        if len(lm_list) != 0:
            fingers = detector.fingers_up()

            # Volume control active when Index finger is extended
            if len(fingers) >= 2 and fingers[1] == 1:
                # Measure distance between Thumb tip (4) and Index tip (8)
                length, img, lineInfo = detector.find_distance(4, 8, img, r=8, t=2)

                # Map distance [25, 180] pixels to Volume [0, 100]%
                volPer = np.interp(length, [25, 180], [0, 100])
                volBar = np.interp(length, [25, 180], [400, 150])

                # Visual indicator when pinched close (min volume / mute threshold)
                if length < 25:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 12, (0, 255, 0), cv2.FILLED)

                # Adjust system volume with pyautogui
                diff = int(volPer - previous_vol)
                if diff < -1:
                    steps = abs(diff) // 2 or 1
                    for _ in range(steps):
                        pyautogui.press('volumedown')
                    previous_vol = int(volPer)
                elif diff > 1:
                    steps = diff // 2 or 1
                    for _ in range(steps):
                        pyautogui.press('volumeup')
                    previous_vol = int(volPer)

        # 4. Draw on-screen Volume HUD Bar
        cv2.rectangle(img, (40, 150), (75, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (40, int(volBar)), (75, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (30, 440), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(img, "Volume Control", (40, 130), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)

        # 5. FPS calculation and display
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # 6. Display window
        cv2.imshow("AI Gesture Volume Controller", img)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

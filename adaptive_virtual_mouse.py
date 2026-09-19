"""
adaptive_virtual_mouse.py
Real-time AI Virtual Mouse driven by hand gestures using MediaPipe Tasks Vision and OpenCV.
Supports cursor movement, left clicks, right clicks, and drive shortcuts.
Author: TheMalyala
"""

import os
import time
import cv2
import numpy as np
import autopy
import hand_tracker as ht


def main():
    wCam, hCam = 640, 480
    frameR = 100  # Frame Reduction boundary
    smoothening = 7

    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    detector = ht.HandDetector(max_hands=1)
    screen_w, screen_h = autopy.screen.size()
    wScr, hScr = int(screen_w), int(screen_h)

    print("AI Adaptive Virtual Mouse started. Press ESC in the camera window to exit.")

    while True:
        # 1. Grab camera frame
        success, img = cap.read()
        if not success or img is None:
            continue

        # 2. Detect hand landmarks
        img = detector.find_hands(img)
        lm_list, bbox = detector.find_position(img)

        # 3. Process gesture points
        if len(lm_list) != 0:
            x1, y1 = lm_list[8][1:]   # Index tip
            x2, y2 = lm_list[12][1:]  # Middle tip

            fingers = detector.fingers_up()
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

            # Mode A: Only Index finger up -> Mouse Moving Mode
            if len(fingers) >= 3 and fingers[1] == 1 and fingers[2] == 0:
                # Map camera frame coordinates to screen resolution
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                # Apply motion smoothening
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # Clamp coordinates to screen boundaries to prevent out-of-bounds exceptions
                targetX = float(np.clip(wScr - clocX, 0, wScr - 1))
                targetY = float(np.clip(clocY, 0, hScr - 1))
                autopy.mouse.move(targetX, targetY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            # Mode B: Both Index and Middle fingers up -> Clicking Modes
            if len(fingers) >= 3 and fingers[1] == 1 and fingers[2] == 1:
                # Left Click: Index tip (8) and Middle tip (12) pinched close
                length, img, lineInfo = detector.find_distance(8, 12, img)
                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click()

                # Right Click: Thumb (4) and Pinky (20) pinched close
                length_rc, img, lineInfo_rc = detector.find_distance(4, 20, img)
                if length_rc < 40:
                    cv2.circle(img, (lineInfo_rc[4], lineInfo_rc[5]), 15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click(button=autopy.mouse.Button.RIGHT)

                # Drive Shortcut: Thumb (4) and Ring (16) pinched close
                length_od, img, lineInfo_od = detector.find_distance(4, 16, img)
                if length_od < 40:
                    cv2.circle(img, (lineInfo_od[4], lineInfo_od[5]), 15, (0, 255, 0), cv2.FILLED)
                    try:
                        os.startfile("A:\\")
                    except Exception:
                        pass

        # 4. FPS counter calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # 5. Display output
        cv2.imshow("AI Adaptive Virtual Mouse", img)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

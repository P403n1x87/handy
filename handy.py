from math import sqrt

import cv2
import mediapipe as mp
from pynput.mouse import Controller


import tkinter as tk

root = tk.Tk()

W = root.winfo_screenwidth()
H = root.winfo_screenheight()


mouse = Controller()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
HL = mp.solutions.hands.HandLandmark

# for _ in mp_hands.HAND_CONNECTIONS:
#     print(_)


def dist(lm1, lm2):
    dx = lm1.x - lm2.x
    dy = lm1.y - lm2.y

    return sqrt(dx * dx + dy * dy)


# For webcam input:
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(-1)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[HL.WRIST]
            index_tip = hand_landmarks.landmark[HL.INDEX_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[HL.PINKY_TIP]
            middle_tip = hand_landmarks.landmark[HL.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[HL.RING_FINGER_TIP]

            mouse.position = (W * (wrist.x-0.2) * 1.5, H * (wrist.y-0.2) * 1.5)

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow("MediaPipe Hands", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
hands.close()
cap.release()

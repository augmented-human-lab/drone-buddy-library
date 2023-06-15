import time

import cv2
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

ud = 0
fb = 0
p_time = 0


def count_fingers(frame, show_feedback=False):
    global p_time

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(frame, f'FPS:{int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    finger_count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 12:
                    cv2.circle(frame, (cx, cy), 10, (255, 255, 0), cv2.FILLED)
                    cv2.putText(frame, f'id12z:{int(lm.z * (-1000))}', (50, 250), cv2.FONT_HERSHEY_PLAIN, 3,
                                (255, 0, 0),
                                3)
                mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
            hand_index = results.multi_hand_landmarks.index(hand_landmarks)
            hand_label = results.multi_handedness[hand_index].classification[0].label
            hand_landmarks = []
            for landmarks in hand_landmarks.landmark:
                hand_landmarks.append([landmarks.x, landmarks.y])
            if hand_label == "Left" and hand_landmarks[4][0] > hand_landmarks[3][0]:
                finger_count = finger_count + 1
            elif hand_label == "Right" and hand_landmarks[4][0] < hand_landmarks[3][0]:
                finger_count = finger_count + 1
            if hand_landmarks[8][1] < hand_landmarks[6][1]:
                finger_count = finger_count + 1
            if hand_landmarks[12][1] < hand_landmarks[10][1]:
                finger_count = finger_count + 1
            if hand_landmarks[16][1] < hand_landmarks[14][1]:
                finger_count = finger_count + 1
            if hand_landmarks[20][1] < hand_landmarks[18][1]:
                finger_count = finger_count + 1
    cv2.rectangle(frame, (390, 10), (470, 120), (0, 255, 255), cv2.FILLED)
    cv2.putText(frame, str(finger_count), (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

    if show_feedback:
        cv2.imshow("Image", frame)

    return finger_count

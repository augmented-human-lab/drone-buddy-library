import time

import cv2
import mediapipe as mp
from djitellopy import Tello

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

drone = Tello()
drone.connect()
drone.streamon()
drone.takeoff()
drone.move_up(100)
ud = 0
fb = 0
pTime = 0

while True:
    drone.send_rc_control(0, fb, ud, 0)
    img = drone.get_frame_read().frame
    img = cv2.resize(img, (500, 500))

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)

    fingerCount = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 12:
                    cv2.circle(img, (cx, cy), 10, (255, 255, 0), cv2.FILLED)
                    cv2.putText(img, f'id12z:{int(lm.z * (-1000))}', (50, 250), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0),
                                3)
                mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)
            handIndex = results.multi_hand_landmarks.index(hand_landmarks)
            handLabel = results.multi_handedness[handIndex].classification[0].label
            handLandmarks = []
            for landmarks in hand_landmarks.landmark:
                handLandmarks.append([landmarks.x, landmarks.y])
            if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                fingerCount = fingerCount + 1
            elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                fingerCount = fingerCount + 1
            if handLandmarks[8][1] < handLandmarks[6][1]:
                fingerCount = fingerCount + 1
            if handLandmarks[12][1] < handLandmarks[10][1]:
                fingerCount = fingerCount + 1
            if handLandmarks[16][1] < handLandmarks[14][1]:
                fingerCount = fingerCount + 1
            if handLandmarks[20][1] < handLandmarks[18][1]:
                fingerCount = fingerCount + 1
    cv2.rectangle(img, (390, 10), (470, 120), (0, 255, 255), cv2.FILLED)
    cv2.putText(img, str(fingerCount), (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
    if fingerCount == 0:
        fb, ud = 0, 0
    if fingerCount == 1:
        fb, ud = 0, 15
    if fingerCount == 2:
        fb, ud = 0, -15
    if fingerCount == 3:
        fb, ud = 15, 0
    if fingerCount == 4:
        fb, ud = -15, 0
    if fingerCount == 5:
        fb, ud = 0, 0
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

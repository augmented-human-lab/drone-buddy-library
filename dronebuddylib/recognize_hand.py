# import time
#
# import cv2
# import mediapipe as mp
#
#
# class MyMediaPipe:
#     mpHands = mp.solutions.hands
#     hands = mpHands.Hands()
#     mpDraw = mp.solutions.drawing_utils
#
#     def get_hands(self) -> mpHands:
#         return self.hands
#
#     def get_mp_draw(self) -> mpDraw:
#         return self.mpDraw
#
#
# def init_mediapipe():
#
#
# def format_image(image):
#     image = cv2.resize(image, (500, 500))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     pTime = 0
#
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime
#
#     cv2.putText(image, f'FPS:{int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
#
#     imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     return image
#
# # def get_finger_count(image, media_pipe_class: MyMediaPipe):
# #     results = hands.process(image)
# #     for hand_landmarks in results.multi_hand_landmarks:
# #         for id, lm in enumerate(hand_landmarks.landmark):
# #             h, w, c = image.shape
# #             cx, cy = int(lm.x * w), int(lm.y * h)
# #             if id == 12:
# #                 cv2.circle(image, (cx, cy), 10, (255, 255, 0), cv2.FILLED)
# #                 cv2.putText(image, f'id12z:{int(lm.z * (-1000))}', (50, 250), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0),
# #                             3)
# #             mpDraw.draw_landmarks(image, hand_landmarks, mpHands.HAND_CONNECTIONS)
# #         handIndex = results.multi_hand_landmarks.index(hand_landmarks)
# #         handLabel = results.multi_handedness[handIndex].classification[0].label
# #         handLandmarks = []
# #         for landmarks in hand_landmarks.landmark:
# #             handLandmarks.append([landmarks.x, landmarks.y])
# #         if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
# #             fingerCount = fingerCount + 1
# #         elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
# #             fingerCount = fingerCount + 1
# #         if handLandmarks[8][1] < handLandmarks[6][1]:
# #             fingerCount = fingerCount + 1
# #         if handLandmarks[12][1] < handLandmarks[10][1]:
# #             fingerCount = fingerCount + 1
# #         if handLandmarks[16][1] < handLandmarks[14][1]:
# #             fingerCount = fingerCount + 1
# #         if handLandmarks[20][1] < handLandmarks[18][1]:
# #             fingerCount = fingerCount + 1
# #     cv2.rectangle(image, (390, 10), (470, 120), (0, 255, 255), cv2.FILLED)
# #     cv2.putText(image, str(fingerCount), (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
# #
# #

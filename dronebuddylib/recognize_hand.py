import cv2
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

def format_image(image):
    image = cv2.resize(image, (500, 500))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
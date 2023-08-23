import cv2
import mediapipe as mp


def get_hand_landmark(img: list):
    """
    Detect hands in an image.

    Args:
        img (list): the frame to detect the hand in

    Returns:
        list | bool: return the list of the landmark of one hand in the frame.
        Return false if no hand is detected.
    """
    image = img.copy()
    hand_landmarks = []
    with mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks_ in results.multi_hand_landmarks:
                for landmark in hand_landmarks_.landmark:
                    hand_landmarks.append([landmark.x, landmark.y])
        else:
            hand_landmarks = False
    return hand_landmarks

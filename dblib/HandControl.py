import mediapipe as mp
import cv2


# This is a function to get the finger count based on the input image.
class GestureRecognition:
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands()

    # The function is to count the number of fingers.
    # The input is RGB image, the output is the number.
    def count_finger(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        coordinates = [(8, 6), (12, 10), (16, 14), (20, 18)]
        thumb_coordinate = (4, 2)
        results = self.hands.process(image)
        multiLandMarks = results.multi_hand_landmarks

        count = 0
        point_list = []

        if multiLandMarks:
            for handLms in multiLandMarks:
                self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
                for idx, lm in enumerate(handLms.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    point_list.append((cx, cy))

            # Circle the identified hand point.
            #for point in point_list:
                #cv2.circle(image, point, 10, (255, 255, 0), cv2.FILLED)

        # Count the number of fingers.
            for coordinate in coordinates:
                # If the coordinate of DIP is higher than PIP (y-axis), then the finger is up.
                if point_list[coordinate[0]][1] < point_list[coordinate[1]][1]:
                    count += 1
                # For thumb, we check the x-axis.
            if point_list[thumb_coordinate[0]][0] > point_list[thumb_coordinate[1]][0]:
                count += 1
        return count

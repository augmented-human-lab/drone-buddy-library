import time
import pkg_resources
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import GestureRecognizerOptions, GestureRecognizer, GestureRecognizerResult

from dronebuddylib.atoms.bodyfeatureextraction.i_feature_extraction import IFeatureExtraction
import mediapipe as mp
import cv2

from dronebuddylib.atoms.objectdetection.mp_object_detection_impl import VisionRunningMode
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import Configurations
from dronebuddylib.utils.utils import config_validity_check

# Initialize Mediapipe's hand module for detecting hand landmarks
mpHands = mp.solutions.hands
hands = mpHands.Hands()
# Initialize Mediapipe's drawing utils for drawing hand landmarks on the image
mpDraw = mp.solutions.drawing_utils


class HandFeatureExtractionImpl(IFeatureExtraction):
    """
    Implementation of the hand feature extraction using Mediapipe's hand detection solution.
    """

    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Constructor for HandFeatureExtractionImpl class.

        Args:
            engine_configurations (EngineConfigurations): Configurations for the engine.
        """
        super().__init__()
        self.hand_landmark = None
        self.gesture_recognition_model = None

        # Check if the configurations are valid for the engine
        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())
        self.configs = engine_configurations

    def get_feature(self, image) -> list:
        """
        Detect hands in an image.

        Args:
            image (list): The frame to detect the hand in.

        Returns:
            list | bool: Return the list of the landmark of one hand in the frame.
                         Return False if no hand is detected.
        """
        copied_image = image.copy()
        hand_landmarks = []
        with mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.5) as hands:
            results = hands.process(cv2.cvtColor(copied_image, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks_ in results.multi_hand_landmarks:
                    for landmark in hand_landmarks_.landmark:
                        hand_landmarks.append([landmark.x, landmark.y])
            else:
                hand_landmarks = False

        self.hand_landmark = hand_landmarks
        return hand_landmarks

    def count_fingers(self, frame, show_feedback=False) -> int:
        """
        Count the number of fingers in a frame.

        Args:
            frame (np.array): The frame to count fingers in.
            show_feedback (bool): Whether to show the processed frame.

        Returns:
            int: The number of fingers in the frame.
        """
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

    def get_gesture(self, numpy_image: list) -> GestureRecognizerResult:
        """
        Get the gesture in an image.

        Args:
            numpy_image (list): The image to recognize the gesture in.

        Returns:
            GestureRecognizerResult: The result of gesture recognition.
        """
        if self.configs.get_configuration(Configurations.HAND_FEATURE_EXTRACTION_ENABLE_GESTURE_RECOGNITION) is True:
            if self.configs.get_configuration(
                    Configurations.HAND_FEATURE_EXTRACTION_GESTURE_RECOGNITION_MODEL_PATH) is not None:
                model_path = self.configs.get_configuration(
                    Configurations.HAND_FEATURE_EXTRACTION_GESTURE_RECOGNITION_MODEL_PATH)
            else:
                model_path = pkg_resources.resource_filename(__name__,
                                                             "bodyfeatureextraction/resources/gesture_recognizer.task")
            options = GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE)
            with GestureRecognizer.create_from_options(options) as recognizer:
                self.gesture_recognition_model = recognizer
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
        # Perform gesture recognition on the provided single image.
        # The gesture recognizer must be created with the image mode.
        return self.gesture_recognition_model.recognize(mp_image)

    def get_required_params(self) -> list:
        """
        Get the required parameters for the engine.

        Returns:
            list: The list of required parameters.
        """
        return []

    def get_optional_params(self) -> list:
        """
        Get the optional parameters for the engine.

        Returns:
            list: The list of optional parameters.
        """
        return [Configurations.HAND_FEATURE_EXTRACTION_ENABLE_GESTURE_RECOGNITION,
                Configurations.HAND_FEATURE_EXTRACTION_GESTURE_RECOGNITION_MODEL_PATH]

    def get_class_name(self) -> str:
        """
        Get the class name of the engine.

        Returns:
            str: The class name of the engine.
        """
        return "HAND_FEATURE_EXTRACTION"

    def get_algorithm_name(self) -> str:
        """
        Get the algorithm name of the engine.

        Returns:
            str: The algorithm name of the engine.
        """
        return "Extracts features and executes functions related to hands"

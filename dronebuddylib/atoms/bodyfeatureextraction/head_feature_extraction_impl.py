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

# Mediapipe hands module for detecting hand landmarks
mpHands = mp.solutions.hands
hands = mpHands.Hands()
# Mediapipe drawing utils for drawing hand landmarks on the image
mpDraw = mp.solutions.drawing_utils


class HeadFeatureExtractionImpl(IFeatureExtraction):
    """
    Implementation of the head feature extraction using mediapipe's face detection solution.
    """
    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Constructor for HeadFeatureExtractionImpl class.

        Args:
            engine_configurations (EngineConfigurations): configurations for the engine.
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
        Get the bounding box of the head in front of the drone.

        Args:
            image (np.array): the image to be processed
        Returns:
            List containing the coordinates and dimensions of the bounding box [x, y, w, h]:
            x (int): x coordinate of the left top corner of the bounding box,
            y (int): y coordinate of the left top corner of the bounding box,
            w (int): width of the bounding box,
            h (int): height of the bounding box.
        """
        mp_face_detection = mp.solutions.face_detection
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
            results = face_detection.process(frame_rgb)
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(max(bboxC.ymin - 0.1, 0) * ih), int(bboxC.width * iw), int(
                        (bboxC.ymin + bboxC.height - max(bboxC.ymin - 0.1, 0)) * ih)
                    return [x, y, w, h]

    def get_required_params(self) -> list:
        """
        Get the required parameters for the engine.

        Returns:
            List containing the required parameters.
        """
        return []

    def get_optional_params(self) -> list:
        """
        Get the optional parameters for the engine.

        Returns:
            List containing the optional parameters.
        """
        return []

    def get_class_name(self) -> str:
        """
        Get the class name.

        Returns:
            String containing the class name.
        """
        return "HEAD_FEATURE_EXTRACTION"

    def get_algorithm_name(self) -> str:
        """
        Get the algorithm name.

        Returns:
            String containing the algorithm name.
        """
        return "Extracts features and executes functions related to head"

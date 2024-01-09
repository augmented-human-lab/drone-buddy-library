import pkg_resources
from mediapipe.tasks.python.vision import PoseLandmarkerResult

from dronebuddylib.atoms.bodyfeatureextraction.i_feature_extraction import IFeatureExtraction
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import AtomicEngineConfigurations

import mediapipe as mp

from dronebuddylib.utils.utils import config_validity_check

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class BodyFeatureExtractionImpl(IFeatureExtraction):
    """
    The BodyFeatureExtractionImpl class is used to extract features related to body postures from an image.
    built on top of Mediapipe's pose landmarking solution.
    for more information: https://mediapipe-studio.webapps.google.com/home
    """

    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Constructor for the BodyFeatureExtractionImpl class.

        Args:
            engine_configurations (EngineConfigurations): The engine configurations object.
        """
        super().__init__()
        self.hand_landmark = None
        self.gesture_recognition_model = None

        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())
        self.configs = engine_configurations

    def get_feature(self, image) -> list:
        """
        Abstract method to get features from an image. This method should be implemented by subclasses.

        Args:
            image (list): The image to extract features from.

        Returns:
            list: The extracted features.
        """
        pass

    def get_supported_features(self) -> list:
        """
        Get the list of supported features for the engine.

        Returns:
            list: The list of supported features.
        """
        return ["POSE"]

    def get_detected_pose(self, image) -> PoseLandmarkerResult:
        """
        Get the detected pose from an image.

        Args:
            image : The numpy list image to detect the pose from.

        Returns:
            PoseLandmarkerResult: The detected pose.
        """
        if self.configs.get_configuration(
                AtomicEngineConfigurations.HAND_FEATURE_EXTRACTION_GESTURE_RECOGNITION_MODEL_PATH) is not None:
            model_path = self.configs.get_configuration(
                AtomicEngineConfigurations.HAND_FEATURE_EXTRACTION_GESTURE_RECOGNITION_MODEL_PATH)
        else:
            model_path = pkg_resources.resource_filename(__name__,
                                                         "resources/pose_landmarker_heavy.task")
        # Read the model file
        with open(model_path, 'rb') as file:
            model_data = file.read()

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_data),
            running_mode=VisionRunningMode.IMAGE)

        with PoseLandmarker.create_from_options(options) as landmarker:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            return landmarker.detect(mp_image)

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        """
           Draws pose landmarks on the given RGB image based on the detection result.

           This method takes an RGB image and a detection result containing pose landmarks. It copies the original image and then iterates through each detected pose, drawing the landmarks on the copy of the image. The landmarks are drawn according to the specifications provided in the detection result, which includes the coordinates and connections of each landmark point.

           Args:
               rgb_image (numpy.ndarray): The original RGB image on which landmarks need to be drawn.
               detection_result (object): An object containing the detected pose landmarks. It typically includes a list of pose landmarks with their x, y, z coordinates.

           Returns:
               numpy.ndarray: An annotated image with pose landmarks drawn on it.

           The method utilizes `solutions.drawing_utils.draw_landmarks` for drawing, which requires converting the landmarks into a format compatible with the drawing utility. Each pose landmark is converted into a `NormalizedLandmark` and then drawn on the image using the specified pose connections and drawing style.
           """
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image

    def get_required_params(self) -> list:
        """
        Get the required parameters for the class.

        Returns:
            list: The list of required parameters.
        """
        return []

    def get_optional_params(self) -> list:
        """
        Get the optional parameters for the class.

        Returns:
            list: The list of optional parameters.
        """
        return [AtomicEngineConfigurations.BODY_FEATURE_EXTRACTION_POSTURE_DETECTION_MODEL_PATH]

    def get_class_name(self) -> str:
        """
        Get the class name.

        Returns:
            str: The class name.
        """
        return "BODY_FEATURE_EXTRACTION"

    def get_algorithm_name(self) -> str:
        """
        Get the algorithm name.

        Returns:
            str: The algorithm name.
        """
        return "Body Feature Extraction"

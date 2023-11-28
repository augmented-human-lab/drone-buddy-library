import pkg_resources
from mediapipe.tasks.python.vision import PoseLandmarkerResult

from dronebuddylib.atoms.bodyfeatureextraction.i_feature_extraction import IFeatureExtraction
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import Configurations

import mediapipe as mp

from dronebuddylib.utils.utils import config_validity_check

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
            image (list): The numpy list image to detect the pose from.

        Returns:
            PoseLandmarkerResult: The detected pose.
        """
        if self.configs.get_configuration(Configurations.HAND_FEATURE_EXTRACTION_ENABLE_GESTURE_RECOGNITION) is True:
            if self.configs.get_configuration(
                    Configurations.HAND_FEATURE_EXTRACTION_GESTURE_RECOGNITION_MODEL_PATH) is not None:
                model_path = self.configs.get_configuration(
                    Configurations.HAND_FEATURE_EXTRACTION_GESTURE_RECOGNITION_MODEL_PATH)
            else:
                model_path = pkg_resources.resource_filename(__name__,
                                                             "bodyfeatureextraction/resources/posture_landmarker_heavy.task")

            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE)

            with PoseLandmarker.create_from_options(options) as landmarker:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                return landmarker.detect(mp_image)

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
        return [Configurations.BODY_FEATURE_EXTRACTION_POSTURE_DETECTION_MODEL_PATH]

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

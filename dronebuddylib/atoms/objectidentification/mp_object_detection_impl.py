import pkg_resources
from mediapipe.tasks.python.vision import ObjectDetectorOptions

from dronebuddylib.atoms.objectdetection.i_object_detection import IObjectDetection

import mediapipe as mp

from dronebuddylib.models.enums import AtomicEngineConfigurations
from dronebuddylib.atoms.objectdetection.detected_object import DetectedObject, BoundingBox, ObjectDetectionResult
from dronebuddylib.utils.logger import Logger
from dronebuddylib.utils.utils import config_validity_check

from mediapipe.tasks.python import vision, BaseOptions

from dronebuddylib.models.engine_configurations import EngineConfigurations

VisionRunningMode = mp.tasks.vision.RunningMode

logger = Logger()


class MPObjectDetectionImpl(IObjectDetection):
    def get_class_name(self) -> str:
        """
        Returns the class name.

        Returns:
            str: The class name.
        """
        return 'OBJECT_DETECTION_MP'

    def get_algorithm_name(self) -> str:
        """
        Returns the algorithm name.

        Returns:
            str: The algorithm name.
        """
        return 'Mediapipe Object Detection'

    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Initializes the object detection engine with the given configurations.

        Args:
            engine_configurations (EngineConfigurations): The engine configurations.
        """
        super().__init__(engine_configurations)
        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())

        path = pkg_resources.resource_filename(__name__, "resources/efficientdet_lite0.tflite")

        configs = engine_configurations.get_configurations_for_engine(self.get_class_name())
        model_path = configs.get(AtomicEngineConfigurations.OBJECT_DETECTION_MP_MODELS_PATH)
        if model_path is not None:
            path = model_path

        # Read the model file
        with open(path, 'rb') as file:
            model_data = file.read()
        logger.log_debug(self.get_class_name() ,'Initializing with model data from ' + path + '')

        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_buffer=model_data),
            max_results=5,
            running_mode=VisionRunningMode.IMAGE)

        detector = vision.ObjectDetector.create_from_options(options)
        self.detector = detector
        logger.log_debug(self.get_class_name() ,'Completed initializing the mediapipe object detection')

    def get_detected_objects(self, image) -> ObjectDetectionResult:
        """
        Detects objects in the provided image and returns a result containing a list of detected objects.

        Args:
            image: The image in which to detect objects.

        Returns:
            ObjectDetectionResult (ObjectDetectionResult) : A result containing a list of detected objects.
        """
        logger.log_debug(self.get_class_name() ,': Detection started.')

        detection_result = self.detector.detect(image)
        logger.log_debug(self.get_class_name() ,'Detection Successful.')
        return_list = []
        simple_list = []
        for detected_object in detection_result.detections:
            formatted_object = DetectedObject([], BoundingBox(detected_object.bounding_box.origin_x,
                                                              detected_object.bounding_box.origin_y,
                                                              detected_object.bounding_box.width,
                                                              detected_object.bounding_box.height))
            highest_confidence = 0
            highest_confident_category = ""
            for label in detected_object.categories:
                formatted_object.add_category(label.category_name, label.score)
                if label.score > highest_confidence:
                    highest_confidence = label.score
                    formatted_object.add_category(label.category_name, label.score)
                    highest_confident_category = label.category_name

            simple_list.append(highest_confident_category)
            return_list.append(formatted_object)

        logger.log_debug(self.get_class_name() ,'Detection completed.')

        return ObjectDetectionResult(simple_list, return_list)

    def get_bounding_boxes_of_detected_objects(self, image) -> list:
        """
        Detects objects in the provided image and returns a list of bounding boxes for the detected objects.

        Args:
            image: The image in which to detect objects.

        Returns:
            list: A list of bounding boxes for the detected objects.
        """
        pass

    def get_required_params(self) -> list:
        """
        Returns a list of required configuration parameters.

        Returns:
            list: A list of required configuration parameters.
        """
        return []

    def get_optional_params(self) -> list:
        """
        Returns a list of optional configuration parameters.

        Returns:
            list: A list of optional configuration parameters.
        """
        return [AtomicEngineConfigurations.OBJECT_DETECTION_MP_MODELS_PATH]

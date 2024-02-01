from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import VisionAlgorithm
from dronebuddylib.atoms.objectdetection.detected_object import ObjectDetectionResult
from dronebuddylib.utils.logger import Logger

logger = Logger()


class ObjectDetectionEngine:

    def __init__(self, algorithm: VisionAlgorithm, config: EngineConfigurations):
        """
        Initializes the object detection engine with the specified algorithm and configuration.

        Args:
            algorithm (VisionAlgorithm): The vision algorithm to be used for object detection.
            config (EngineConfigurations): The configuration for the vision engine.
        """
        if algorithm == VisionAlgorithm.YOLO or algorithm == VisionAlgorithm.YOLO.name:
            logger.log_info(self.get_class_name(), 'Preparing to initialize YOLO object detection engine.')
            from dronebuddylib.atoms.objectdetection.yolo_object_detection_impl import YOLOObjectDetectionImpl
            self.vision_engine = YOLOObjectDetectionImpl(config)
        elif algorithm == VisionAlgorithm.MEDIA_PIPE or algorithm == VisionAlgorithm.MEDIA_PIPE.name:
            logger.log_info(self.get_class_name(), 'Preparing to initialize Mediapipe object detection engine.')
            from dronebuddylib.atoms.objectdetection.mp_object_detection_impl import MPObjectDetectionImpl
            self.vision_engine = MPObjectDetectionImpl(config)

    def get_class_name(self) -> str:
        """
        Returns the class name.

        Returns:
            str: The class name.
        """
        return 'OBJECT_DETECTION_ENGINE'

    def get_detected_objects(self, frame) -> ObjectDetectionResult:
        """
        Detects objects in a given frame using the specified vision algorithm.

        Args:
            frame: The input frame for which objects need to be detected.

        Returns:
            ObjectDetectionResult (ObjectDetectionResult): The result of the object detection, including a list of detected objects.
        """
        return self.vision_engine.get_detected_objects(frame)

    def get_bounding_boxes_of_detected_objects(self, frame) -> list:
        """
        Retrieves bounding boxes for objects in a given frame using the specified vision algorithm.

        Args:
            frame: The input frame for which bounding boxes are to be retrieved.

        Returns:
            list: A list of bounding boxes for the detected objects.
        """
        return self.vision_engine.get_bounding_boxes_of_detected_objects(frame)


from dronebuddylib.atoms.objectdetection.mp_object_detection_impl import MPObjectDetectionImpl
from dronebuddylib.atoms.objectdetection.yolo_object_detection_impl import YOLOObjectDetectionImpl
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import VisionAlgorithm
from dronebuddylib.models.object_detected import ObjectDetectionResult


class ObjectDetectionEngine:

    def __init__(self, algorithm: VisionAlgorithm, config: EngineConfigurations):

        if algorithm == VisionAlgorithm.YOLO:
            self.vision_engine = YOLOObjectDetectionImpl(config)
        if algorithm == VisionAlgorithm.MEDIA_PIPE:
            self.vision_engine = MPObjectDetectionImpl(config)

    def get_detected_objects(self, frame) -> ObjectDetectionResult:
        """
        Detects objects in a given frame using the specified vision algorithm.

        Parameters:
        - algorithm (VisionAlgorithm): The vision algorithm to be used for object detection.
        - vision_config (VisionConfigs): Configuration for the vision algorithm, including weights path.
        - frame: The input frame for which objects need to be detected.

        Returns:
        - list: List of detected objects if using YOLO V8.

        Note:
        Only YOLO V8 and Media pipe is implemented as of now.
        """
        return self.vision_engine.get_detected_objects(frame)

    def get_bounding_boxes_of_detected_objects(self, frame) -> list:
        """
        Retrieves bounding boxes for objects in a given frame using the specified vision algorithm.

        Parameters:
        - frame: The input frame for which bounding boxes are to be retrieved.

        Returns:
        - list: List of bounding boxes if using YOLO V8.

        Note:
            Only YOLO V8 and Media pipe is implemented as of now.
        """
        return self.vision_engine.get_bounding_boxes_of_detected_objects(frame)

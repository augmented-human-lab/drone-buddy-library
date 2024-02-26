from ultralytics import YOLO

from dronebuddylib.atoms.objectdetection.i_object_detection import IObjectDetection
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import AtomicEngineConfigurations
from dronebuddylib.atoms.objectdetection.detected_object import DetectedObject, BoundingBox, ObjectDetectionResult
from dronebuddylib.utils.logger import Logger
from dronebuddylib.utils.utils import config_validity_check

logger = Logger()


class YOLOObjectDetectionImpl(IObjectDetection):
    def get_class_name(self) -> str:
        """
        Gets the class name of the object detection implementation.

        Returns:
            str: The class name of the object detection implementation.
        """
        return 'OBJECT_DETECTION_YOLO'

    def get_algorithm_name(self) -> str:
        """
        Gets the algorithm name of the object detection implementation.

        Returns:
            str: The algorithm name of the object detection implementation.
        """
        return 'YOLO V8 Object Detection'

    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Initializes the YOLO V8 object detection engine with the given engine configurations.

        Args:
            engine_configurations (EngineConfigurations): The engine configurations for the object detection engine.
        """
        super().__init__(engine_configurations)
        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())

        configs = engine_configurations.get_configurations_for_engine(self.get_class_name())
        model_name = configs.get(AtomicEngineConfigurations.OBJECT_DETECTION_YOLO_VERSION)
        if model_name is None:
            model_name = "yolov8n.pt"

        logger.log_info(self.get_class_name(), ':Initializing with model with ' + model_name + '')

        self.detector = YOLO(model_name)
        self.object_names = self.detector.names
        logger.log_debug(self.get_class_name(), 'Initialized the YOLO object detection')

    def get_detected_objects(self, image) -> ObjectDetectionResult:
        """
        Detects objects in the given image using YOLO V8 object detection engine.

        Args:
            image: The image to detect objects in.

        Returns:
            ObjectDetectionResult (ObjectDetectionResult): The result of the object detection, including a list of detected objects.
        """
        logger.log_debug(self.get_class_name(), 'Detection started.')

        results = self.detector.predict(source=image, save=True, save_txt=True)
        logger.log_debug(self.get_class_name(), ' :Detection Successful.')

        detected_objects = []
        detected_names = []
        # Save predictions as labels
        for result in results:
            for idx, res in enumerate(result.boxes.cls):
                box_c = result.boxes.xywh.tolist()[idx]
                conf_c = result.boxes.conf.tolist()[idx]
                detected = DetectedObject([], BoundingBox(box_c[0], box_c[1], box_c[2], box_c[3]))
                detected.add_category(self.object_names[int(res)], conf_c)
                detected_objects.append(detected)
                detected_names.append(self.object_names[int(res)])
                logger.log_debug(self.get_class_name(), 'Detection completed.')

        return ObjectDetectionResult(detected_names, detected_objects)

    def get_bounding_boxes_of_detected_objects(self, image) -> list:
        """
        Gets the bounding boxes of objects detected in an image using YOLO V8 object detection engine.

        Args:
            image: The image to detect objects in.

        Returns:
            list: A list of bounding boxes corresponding to the objects detected in the image.
        """
        # Additional logic for bounding boxes can be implemented here
        return []

    def get_required_params(self) -> list:
        """
        Gets the list of required configuration parameters for YOLO V8 object detection engine.

        Returns:
            list: The list of required configuration parameters.
        """
        return [AtomicEngineConfigurations.OBJECT_DETECTION_YOLO_VERSION]

    def get_optional_params(self) -> list:
        """
        Gets the list of optional configuration parameters for YOLO V8 object detection engine.

        Returns:
            list: The list of optional configuration parameters.
        """
        # Additional optional parameters can be added here
        return []

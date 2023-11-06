import logging
import numpy as np
import pkg_resources
from ultralytics import YOLO
from PIL import Image
import cv2

from dronebuddylib.atoms.objectdetection.i_object_detection import IObjectDetection
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import Configurations
from dronebuddylib.models.object_detected import ObjectDetected, BoundingBox, ObjectDetectionResult
from dronebuddylib.utils.utils import config_validity_check


class YOLOObjectDetectionImpl(IObjectDetection):
    def get_class_name(self) -> str:
        """
        Gets the class name of the object detection implementation.

        Returns:
            str: The class name of the object detection implementation.
        """
        return 'OBJECT_DETECTION_YOLO_V8'

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
        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())

        configs = engine_configurations.get_configurations_for_engine(self.get_class_name())
        model_name = configs.get(Configurations.OBJECT_DETECTION_YOLO_VERSION)
        if model_name is None:
            model_name = "yolov8n.pt"
        self.detector = YOLO(model_name)
        self.object_names = self.detector.names

    def get_detected_objects(self, image) -> ObjectDetectionResult:
        """
        Detects objects in the given image using YOLO V8 object detection engine.

        Args:
            image: The image to detect objects in.

        Returns:
            ObjectDetectionResult: The result of the object detection, including a list of detected objects.
        """
        results = self.detector.predict(source=image, save=True, save_txt=True)
        detected_objects = []
        detected_names = []
        # Save predictions as labels
        for result in results:
            for res in result.boxes.cls:
                detected = ObjectDetected([], BoundingBox(0, 0, 0, 0))
                detected.add_category(self.object_names[int(res)], 0.0)
                detected_objects.append(detected)
                detected_names.append(self.object_names[int(res)])

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
        return [Configurations.OBJECT_DETECTION_YOLO_VERSION]

    def get_optional_params(self) -> list:
        """
        Gets the list of optional configuration parameters for YOLO V8 object detection engine.

        Returns:
            list: The list of optional configuration parameters.
        """
        # Additional optional parameters can be added here
        return []

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
        return 'OBJECT_DETECTION_YOLO_V8'

    def get_algorithm_name(self) -> str:
        return 'YOLO V8 Object Detection'

    def __init__(self, engine_configurations: EngineConfigurations):
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
        results = self.detector.predict(source=image, save=True, save_txt=True)
        detected_objects = []
        detected_names = []
        # save predictions as labels
        for result in results:
            for res in result.boxes.cls:
                detected = ObjectDetected([], BoundingBox(0, 0, 0, 0))
                detected.add_category(self.object_names[int(res)], 0.0)
                detected_objects.append(detected)
                detected_names.append(self.object_names[int(res)])

        return ObjectDetectionResult(detected_names, detected_objects)

    def get_bounding_boxes_of_detected_objects(self, image) -> list:
        """
                Get the bounding boxes of objects detected in an image using a YOLO (You Only Look Once) object detection engine.

                Args:
                    yolo_engine: The YoloEngine object used for object detection.
                    image: The image to detect objects in.

                Returns:
                    A list of bounding boxes corresponding to the objects detected in the image.

                """

        # Get the width and height of the image

        return []

    def get_required_params(self) -> list:
        return [Configurations.OBJECT_DETECTION_YOLO_VERSION]

    def get_optional_params(self) -> list:
        pass

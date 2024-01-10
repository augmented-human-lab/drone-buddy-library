from abc import abstractmethod

from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.i_dbl_function import IDBLFunction
from dronebuddylib.atoms.objectdetection.detected_object import ObjectDetectionResult


class IObjectDetection(IDBLFunction):
    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Constructor to initialize the object detection engine.

        Args:
            engine_configurations (EngineConfigurations): The configurations for the object detection engine.
        """
        self.engine_configurations = engine_configurations

    @abstractmethod
    def get_detected_objects(self, image) -> ObjectDetectionResult:
        """
        Detects objects in the provided image and returns a list of detected objects.

        Args:
            image: The image in which to detect objects.

        Returns:
            ObjectDetectionResult: A result containing a list of detected objects.
        """
        pass

    @abstractmethod
    def get_bounding_boxes_of_detected_objects(self, image) -> list:
        """
        Detects objects in the provided image and returns a list of bounding boxes for the detected objects.

        Args:
            image: The image in which to detect objects.

        Returns:
            list: A list of bounding boxes for the detected objects.
        """
        pass

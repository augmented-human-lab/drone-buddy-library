from abc import abstractmethod

from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.i_dbl_function import IDBLFunction
from dronebuddylib.atoms.objectdetection.detected_object import ObjectDetectionResult


class IObjectRecognition(IDBLFunction):
    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Constructor to initialize the object detection engine.

        Args:
            engine_configurations (EngineConfigurations): The configurations for the object detection engine.
        """
        self.engine_configurations = engine_configurations

    # @abstractmethod
    # def get_detected_objects(self, image) -> ObjectDetectionResult:
    #     """
    #     Detects objects in the provided image and returns a list of detected objects.
    #
    #     Args:
    #         image: The image in which to detect objects.
    #
    #     Returns:
    #         ObjectDetectionResult: A result containing a list of detected objects.
    #     """
    #     pass

    @abstractmethod
    def remember_object(self, object_name, image=None, image_folder_path=None, ) -> bool:
        """
        Remembers an object by associating it with a name.

        Args:
            image: The image containing the object.
            object_name (str): The name to be associated with the object.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        pass

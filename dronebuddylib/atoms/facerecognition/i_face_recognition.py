from abc import abstractmethod

from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.i_dbl_function import IDBLFunction


class IFaceRecognition(IDBLFunction):
    """
    Interface for face recognition functionality.

    Methods:
        recognize_face: Recognizes faces in an image.
        remember_face: Associates a name with a face in an image.
    """
    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Initialize the IFaceRecognition interface.

        Args:
            engine_configurations (EngineConfigurations): The configurations for the engine.
        """
        self.engine_configurations = engine_configurations

    @abstractmethod
    def recognize_face(self, image) -> list:
        """
        Recognize faces in an image.

        Args:
            image: The image containing faces to be recognized.

        Returns:
            list: A list of recognized faces.
        """
        pass

    @abstractmethod
    def remember_face(self, image, name) -> bool:
        """
        Associate a name with a face in an image.

        Args:
            image: The image containing the face to be associated with a name.
            name (str): The name to be associated with the face.

        Returns:
            bool: True if the association was successful, False otherwise.
        """
        pass

from abc import abstractmethod

from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.i_dbl_function import IDBLFunction


class ITextRecognition(IDBLFunction):
    """
    Interface for face recognition functionality.

    Methods:
        recognize_face: Recognizes faces in an image.
        remember_face: Associates a name with a face in an image.
    """

    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Initialize the ITextRecognition interface.

        Args:
            engine_configurations (EngineConfigurations): The configurations for the engine.
        """
        self.engine_configurations = engine_configurations

    @abstractmethod
    def recognize_text(self, image) -> list:
        """
          Recognize text in an image.

          Args:
              image: The image containing text to be recognized.
        """

    pass

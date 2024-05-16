from abc import abstractmethod

from dronebuddylib.atoms.placerecognition.place_recognition_result import RecognizedPlaces
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.i_dbl_function import IDBLFunction


class IPlaceRecognition(IDBLFunction):
    """
    Interface for place recognition functionality.

    This interface defines the methods required for recognizing and remembering places using images. It allows for the association of names with recognized places and testing the memory of the recognition system.

    Methods:
        recognize_place: Recognizes places in an image.
        remember_place: Associates a name with a place in an image.
        test_memory: Tests the accuracy of the place recognition algorithm.
        get_current_status: Retrieves the current status of the place recognition engine.
    """

    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Initialize the IPlaceRecognition interface.

        Args:
            engine_configurations (EngineConfigurations): The configurations for the recognition engine, detailing how the place recognition should be performed.
        """
        self.engine_configurations = engine_configurations

    @abstractmethod
    def recognize_place(self, image) -> RecognizedPlaces:
        """
        Recognize places in an image.

        Args:
            image: The image containing places to be recognized.

        Returns:
            RecognizedPlaces: An object containing a list of recognized places with their details (e.g., names, bounding boxes).
        """
        pass

    @abstractmethod
    def remember_place(self, image=None, name=None) -> bool:
        """
        Associate a name with a place in an image.

        This method is used to remember a place by associating it with a given name, facilitating easier identification in future recognitions.

        Args:
            image: The image containing the place to be associated with a name.
            name (str): The name to be associated with the place.

        Returns:
            bool: True if the association was successful, False otherwise.
        """
        pass

    @abstractmethod
    def create_memory(self):
        """
        Create a memory database or structure for the place recognition engine.

        This method creates a memory structure or database for the place recognition engine, which can be used to optimize future recognition tasks.

        Returns:
            A data structure or system representing the memory of the place recognition engine, useful for improving recognition efficiency and accuracy.
        """
        pass

    @abstractmethod
    def test_memory(self) -> dict:
        """
        Test the accuracy of the place recognition algorithm.

        This method evaluates the accuracy and performance of the place recognition algorithm by returning relevant metrics.

        Returns:
            dict: A dictionary with accuracy-related numbers for the place recognition algorithm, including precision, recall, and F1 score, among others.
        """
        pass

    @abstractmethod
    def get_current_status(self):
        """
        Get the current status of the place recognition engine.

        This method provides the current operational status of the recognition engine, including any errors, operational mode, or other relevant status information.

        Returns:
            A description or object representing the current status of the place recognition engine. The exact return type and structure can be defined as needed.
        """
        pass

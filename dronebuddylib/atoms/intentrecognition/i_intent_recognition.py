from abc import ABC, abstractmethod

from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.i_dbl_function import IDBLFunction


class IIntentRecognition(IDBLFunction, ABC):
    """
    Abstract base class for intent recognition implementations.

    This class defines the interface for intent recognition systems that are used to determine the user's intent based on their input.
    """

    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Initializes the intent recognition system with the given engine configurations.

        Args:
            engine_configurations (EngineConfigurations): The engine configurations to be used by the intent recognition system.
        """
        self.engine_configurations = engine_configurations

    @abstractmethod
    def get_resolved_intent(self, phrase: str) -> str:
        """
        Resolves the user's intent based on the given phrase.

        This method should be implemented by subclasses to provide the functionality for recognizing the user's intent.

        Args:
            phrase (str): The user's input phrase.

        Returns:
            str: The recognized intent.
        """
        pass

    @abstractmethod
    def introduce_new_intents(self, new_intents: dict) -> bool:
        """
        Introduces new intents to the intent recognition system.

        This method should be implemented by subclasses to provide the functionality for adding new intents to the intent recognition system.

        Args:
            new_intents (dict): The new intents to be added.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        pass

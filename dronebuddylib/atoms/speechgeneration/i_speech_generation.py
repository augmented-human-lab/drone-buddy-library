from abc import ABC, abstractmethod

from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.i_dbl_function import IDBLFunction


class ISpeechGeneration(IDBLFunction):
    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Constructor to initialize the speech generation engine.

        Args:
            engine_configurations (EngineConfigurations): The configurations for the speech generation engine.
        """
        self.engine_configurations = engine_configurations

    @abstractmethod
    def read_phrase(self, phrase: str) -> None:
        """
        Generates speech from the provided text and plays it.

        Args:
            phrase (str): The text to be converted into speech.

        Returns:
            None
        """
        pass

    @abstractmethod
    def change_voice(self, voice_id: str) -> bool:
        """
        Changes the voice used by the speech generation engine.

        Args:
            voice_id (str): The identifier of the desired voice.

        Returns:
            bool: True if the voice was changed successfully, False otherwise.
        """
        pass

    @abstractmethod
    def change_volume(self, volume: float) -> bool:
        """
        Changes the volume of the speech generation engine.

        Args:
            volume (float): The desired volume level.

        Returns:
            bool: True if the volume was changed successfully, False otherwise.
        """
        pass

    @abstractmethod
    def change_rate(self, rate: int) -> bool:
        """
        Changes the speech rate of the speech generation engine.

        Args:
            rate (int): The desired speech rate.

        Returns:
            bool: True if the rate was changed successfully, False otherwise.
        """
        pass

    @abstractmethod
    def get_current_configs(self) -> dict:
        """
        Gets the current configurations of the speech generation engine.

        Returns:
            dict: A dictionary containing the current configurations.
        """
        pass

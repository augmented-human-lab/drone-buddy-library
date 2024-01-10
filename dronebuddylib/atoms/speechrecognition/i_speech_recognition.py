from abc import abstractmethod

from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.i_dbl_function import IDBLFunction
from dronebuddylib.atoms.speechrecognition.recognized_speech import RecognizedSpeechResult


class ISpeechRecognition(IDBLFunction):
    """
    This interface defines the methods required for speech-to-text conversion.

    Attributes:
        engine_configurations (EngineConfigurations): The engine configurations containing necessary parameters.
    """
    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Initializes the interface with the provided engine configurations.

        Args:
            engine_configurations (EngineConfigurations): The engine configurations containing necessary parameters.
        """
        self.engine_configurations = engine_configurations

    @abstractmethod
    def recognize_speech(self, audio) -> RecognizedSpeechResult:
        """
        Recognizes speech from an audio input.

        Args:
            audio (bytes): The audio input to be recognized.

        Returns:
            RecognizedSpeechResult: The result containing recognized text and total billed time.
        """
        pass

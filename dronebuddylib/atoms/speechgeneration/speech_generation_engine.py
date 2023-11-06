from dronebuddylib.atoms.speechgeneration.tts_speech_generation_impl import TTSTextToSpeechEngineImpl
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.utils.enums import SpeechGenerationAlgorithm


class SpeechGenerationEngine:
    """
    A facade that provides a simplified interface to the underlying speech generation engines.
    """
    def __init__(self, algorithm: SpeechGenerationAlgorithm, config: EngineConfigurations):
        """
        Initializes the speech generation engine with the specified algorithm and configuration.

        Args:
            algorithm (SpeechGenerationAlgorithm): The speech generation algorithm to be used.
            config (EngineConfigurations): The engine configurations.
        """
        self.algorithm = algorithm
        if algorithm == SpeechGenerationAlgorithm.GOOGLE_TTS_OFFLINE:
            self.speech_generation_engine = TTSTextToSpeechEngineImpl(config)

    def read_phrase(self, phrase: str) -> None:
        """
        Generates speech from the provided phrase.

        Args:
            phrase (str): The phrase to be converted into speech.

        Returns:
            None
        """
        return self.speech_generation_engine.read_phrase(phrase)

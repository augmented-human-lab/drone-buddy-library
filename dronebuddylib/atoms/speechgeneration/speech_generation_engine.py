from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.utils.enums import SpeechGenerationAlgorithm
from dronebuddylib.utils.utils import logger


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
            logger.log_info(self.get_class_name(), 'Preparing to initialize offline Google TTS engine.')
            from dronebuddylib.atoms.speechgeneration.tts_speech_generation_impl import TTSTextToSpeechGenerationImpl
            self.speech_generation_engine = TTSTextToSpeechGenerationImpl(config)

    def get_class_name(self) -> str:
        """
        Returns the class name.

        Returns:
            str: The class name.
        """
        return 'SPEECH_GENERATION_ENGINE'

    def read_phrase(self, phrase: str) -> None:
        """
        Generates speech from the provided phrase.

        Args:
            phrase (str): The phrase to be converted into speech.

        Returns:
            None
        """
        return self.speech_generation_engine.read_phrase(phrase)

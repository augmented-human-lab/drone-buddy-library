from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.utils.enums import SpeechRecognitionAlgorithm
from dronebuddylib.utils.utils import logger


class SpeechRecognitionEngine:
    """
    This class provides a high-level interface for speech to text conversion using different algorithms.

    Attributes:
        algorithm (SpeechRecognitionAlgorithm): The algorithm to be used for speech recognition.
        speech_config (EngineConfigurations): The configurations for the speech recognition engine.
        speech_conversion_engine (ISpeechToTextConversion): The speech recognition engine.
    """

    def __init__(self, algorithm: SpeechRecognitionAlgorithm, speech_config: EngineConfigurations):
        """
        Initializes the SpeechToTextEngine class with the provided algorithm and speech configurations.

        Args:
            algorithm (SpeechRecognitionAlgorithm): The algorithm to be used for speech recognition.
            speech_config (EngineConfigurations): The configurations for the speech recognition engine.
        """
        self.algorithm = algorithm
        self.speech_config = speech_config
        if (algorithm == SpeechRecognitionAlgorithm.GOOGLE_SPEECH_RECOGNITION
                or algorithm == SpeechRecognitionAlgorithm.GOOGLE_SPEECH_RECOGNITION.name):
            logger.log_info(self.get_class_name(), 'Preparing to initialize Google speech recognition engine.')

            from dronebuddylib.atoms.speechrecognition.google_speech_recognition_impl import GoogleSpeechRecognitionImpl

            self.speech_conversion_engine = GoogleSpeechRecognitionImpl(speech_config)

        if (algorithm == SpeechRecognitionAlgorithm.VOSK_SPEECH_RECOGNITION
                or algorithm == SpeechRecognitionAlgorithm.VOSK_SPEECH_RECOGNITION.name):
            logger.log_info(self.get_class_name(), 'Preparing to initialize VOSK speech recognition engine.')

            from dronebuddylib.atoms.speechrecognition.vosk_speech_recognition_impl import VoskSpeechRecognitionImpl

            self.speech_conversion_engine = VoskSpeechRecognitionImpl(speech_config)

        if (algorithm == SpeechRecognitionAlgorithm.MULTI_ALGO_SPEECH_RECOGNITION
                or algorithm == SpeechRecognitionAlgorithm.MULTI_ALGO_SPEECH_RECOGNITION.name):
            logger.log_info(self.get_class_name(), 'Preparing to initialize Multi Algorithm speech recognition engine.')

            from dronebuddylib.atoms.speechrecognition.multi_algo_speech_recognition_impl import \
                MultiAlgoSpeechToTextConversionImplementation
            self.speech_conversion_engine = MultiAlgoSpeechToTextConversionImplementation(speech_config)

    def get_class_name(self) -> str:
        """
        Returns the class name.

        Returns:
            str: The class name.
        """
        return 'SPEECH_RECOGNITION_ENGINE'

    def recognize_speech(self, audio_steam):
        """
        Recognizes speech from an audio stream using the selected speech recognition algorithm.

        Args:
            audio_steam (bytes): The audio stream content to be recognized.

        Returns:
            The result of the speech recognition.
        """
        return self.speech_conversion_engine.recognize_speech(audio_steam)

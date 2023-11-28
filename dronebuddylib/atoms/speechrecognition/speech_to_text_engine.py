from dronebuddylib.atoms.speechrecognition.google_speech_2_text_conversion_impl import GoogleSpeechToTextConversionImpl
from dronebuddylib.atoms.speechrecognition.vosk_speech_2_text_conversion_impl import VoskSpeechToTextConversionImpl
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.utils.enums import SpeechRecognitionAlgorithm


class SpeechToTextEngine:
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
        if algorithm == SpeechRecognitionAlgorithm.GOOGLE_SPEECH_RECOGNITION:
            self.speech_conversion_engine = GoogleSpeechToTextConversionImpl(speech_config)

        if algorithm == SpeechRecognitionAlgorithm.VOSK_SPEECH_RECOGNITION:
            self.speech_conversion_engine = VoskSpeechToTextConversionImpl(speech_config)

    def recognize_speech(self, audio_steam):
        """
        Recognizes speech from an audio stream using the selected speech recognition algorithm.

        Args:
            audio_steam (bytes): The audio stream content to be recognized.

        Returns:
            The result of the speech recognition.
        """
        return self.speech_conversion_engine.recognize_speech(audio_steam)

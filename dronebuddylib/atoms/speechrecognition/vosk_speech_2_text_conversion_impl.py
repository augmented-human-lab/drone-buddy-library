import pkg_resources
from vosk import Model, KaldiRecognizer

from dronebuddylib.atoms.speechrecognition.i_speech_to_text_conversion import ISpeechToTextConversion
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import Configurations
from dronebuddylib.models.recognized_speech import RecognizedSpeechResult
from dronebuddylib.utils.logging_config import get_logger
from dronebuddylib.utils.utils import config_validity_check

# Get an instance of a logger
logger = get_logger()
queue = []

'''
:param language: The language of the model. The default is 'en-US'. (currently only supports this language)
:return: The vosk model.

need to initialize the model before using the speechrecognition to text engine.
'''


class VoskSpeechToTextConversionImpl(ISpeechToTextConversion):
    """
    This class is an implementation of the ISpeechToTextConversion interface for Vosk API.

    Attributes:
        speech_conversion_engine (KaldiRecognizer): The Vosk KaldiRecognizer object for speech recognition.
    """

    def get_class_name(self) -> str:
        """
        Gets the class name.

        Returns:
            str: The class name.
        """
        return 'TEXT_TO_SPEECH_VOSK'

    def get_algorithm_name(self) -> str:
        """
        Gets the algorithm name.

        Returns:
            str: The algorithm name.
        """
        return 'Vosk Text to Speech'

    def get_required_params(self) -> list:
        """
        Gets the list of required parameters.

        Returns:
            list: The list of required parameters.
        """
        return []

    def get_optional_params(self) -> list:
        """
        Gets the list of optional parameters.

        Returns:
            list: The list of optional parameters.
        """
        return [Configurations.SPEECH_RECOGNITION_VOSK_LANGUAGE_MODEL_PATH]

    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Initializes a speech-to-text engine using the Vosk model for a given language.

        Args:
            engine_configurations (EngineConfigurations): The engine configurations containing necessary parameters.
        """
        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())
        configs = engine_configurations.get_configurations_for_engine(self.get_class_name())

        model_path = pkg_resources.resource_filename(__name__, "resources/speechrecognition/vosk-model-small-en-us-0.15")
        language_model_path = configs.get(Configurations.SPEECH_RECOGNITION_VOSK_LANGUAGE_MODEL_PATH, model_path)

        model = Model(language_model_path)
        vosk_kaldi_model = KaldiRecognizer(model, 44100)
        logger.info('Speech Recognition : Initialized speechrecognition recognition model')

        self.speech_conversion_engine = vosk_kaldi_model

    def recognize_speech(self, audio_steam):
        """
        Recognizes text from an audio stream using the Vosk API.

        Args:
            audio_steam (bytes): The audio stream content to be recognized.

        Returns:
            RecognizedSpeechResult: The result containing recognized text and total billed time.
        """
        if self.speech_conversion_engine.AcceptWaveform(audio_steam):
            r = self.speech_conversion_engine.Result()
            logger.debug('Speech Recognition : Recognized utterance : ', r)
            return RecognizedSpeechResult(r, None)
        return None

import json

import pkg_resources
from vosk import Model, KaldiRecognizer

from dronebuddylib.atoms.speechrecognition.i_speech_recognition import ISpeechRecognition
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import AtomicEngineConfigurations
from dronebuddylib.atoms.speechrecognition.recognized_speech import RecognizedSpeechResult
from dronebuddylib.utils.logger import Logger
from dronebuddylib.utils.utils import config_validity_check

# Get an instance of a logger
logger = Logger()
queue = []

'''
:param language: The language of the model. The default is 'en-US'. (currently only supports this language)
:return: The vosk model.

need to initialize the model before using the speechrecognition to text engine.
'''


class VoskSpeechRecognitionImpl(ISpeechRecognition):
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
        return 'SPEECH_RECOGNITION_VOSK'

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
        return [AtomicEngineConfigurations.SPEECH_RECOGNITION_VOSK_LANGUAGE_MODEL_PATH]

    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Initializes a speech-to-text engine using the Vosk model for a given language.

        Args:
            engine_configurations (EngineConfigurations): The engine configurations containing necessary parameters.
        """
        super().__init__(engine_configurations)
        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())
        configs = engine_configurations.get_configurations_for_engine(self.get_class_name())

        model_path = pkg_resources.resource_filename(__name__, "resources/vosk-model-small-en-us-0.15")
        language_model_path = configs.get(AtomicEngineConfigurations.SPEECH_RECOGNITION_VOSK_LANGUAGE_MODEL_PATH,
                                          model_path)

        model = Model(language_model_path)
        logger.log_info(self.get_class_name(), 'Initializing with model with ' + language_model_path + '')

        vosk_kaldi_model = KaldiRecognizer(model, 44100)
        logger.log_info(self.get_class_name(), ' Initialized speechrecognition recognition model')

        self.speech_conversion_engine = vosk_kaldi_model
        logger.log_debug(self.get_class_name(), ' Initialized the Vosk Speech Recognition')

    def recognize_speech(self, audio_steam) -> RecognizedSpeechResult:
        """
        Recognizes text from an audio stream using the Vosk API.

        Args:
            audio_steam (bytes): The audio stream content to be recognized.

        Returns:
            RecognizedSpeechResult: The result containing recognized text and total billed time.
        """
        logger.log_debug(self.get_class_name(), 'Recognition started.')

        if self.speech_conversion_engine.AcceptWaveform(audio_steam):
            r = self.speech_conversion_engine.Result()
            cleaned_string = r.replace("\n", "")
            formatted_string = json.loads(cleaned_string)
            logger.log_success(self.get_class_name(), 'Recognized utterance : ' + r)
            logger.log_debug(self.get_class_name(), 'Recognition Successful.')

            return RecognizedSpeechResult(formatted_string['text'], None)
        return RecognizedSpeechResult("NONE", None)

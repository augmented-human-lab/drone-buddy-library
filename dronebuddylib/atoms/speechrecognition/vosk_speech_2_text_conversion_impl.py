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
    def get_class_name(self) -> str:
        return 'TEXT_TO_SPEECH_VOSK'

    def get_algorithm_name(self) -> str:
        return 'Vosk Text to Speech'

    def get_required_params(self) -> list:
        return []

    def get_optional_params(self) -> list:
        return [Configurations.SPEECH_RECOGNITION_VOSK_LANGUAGE_MODEL_PATH]

    def __init__(self, engine_configurations: EngineConfigurations):
        """
         Initializes a speechrecognition-to-text engine using the Vosk model for a given language.
         (currently only supports 'en-US' language)

         Args:
         - language: a string representing the language code to use (e.g. 'en-us', 'fr-fr')

         Returns:
         - a Vosk KaldiRecognizer object that can be used for speechrecognition recognition
         """
        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())
        configs = engine_configurations.get_configurations_for_engine(self.get_class_name())

        # Define the path to the Vosk model for the given language
        model_path = pkg_resources.resource_filename(__name__,
                                                     "resources/speechrecognition/vosk-model-small-en-us-0.15")
        language_model_path = configs.get(Configurations.SPEECH_RECOGNITION_VOSK_LANGUAGE_MODEL_PATH, model_path)

        # Load the Vosk model and create a KaldiRecognizer object
        model = Model(language_model_path)

        # Log that the speechrecognition recognition model has been initialized
        vosk_kaldi_model = KaldiRecognizer(model, 44100)
        logger.info('Speech Recognition : Initialized speechrecognition recognition model')

        # Return the Vosk KaldiRecognizer object
        self.speech_conversion_engine = vosk_kaldi_model

    def recognize_speech(self, audio_steam):
        """
              Recognizes a text from an audio feed using a given model.

              Args:
              - model: The vosk model that is returned by the init_speech_to_text_engine().
              - audio_feed: a byte string representing the audio feed to recognize, taken by audio_feed.read(num_frames)

              Returns:
              - the text that was recognized, or None if no text was recognized
              """
        if self.speech_conversion_engine.AcceptWaveform(audio_steam):
            r = self.speech_conversion_engine.Result()
            logger.debug('Speech Recognition : Recognized utterance : ', r)
            return RecognizedSpeechResult(r, None)
        return None

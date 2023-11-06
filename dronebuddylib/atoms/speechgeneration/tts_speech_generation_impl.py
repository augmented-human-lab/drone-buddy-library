import pyttsx3

from dronebuddylib.atoms.speechgeneration.i_speech_generation import ISpeechGeneration
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import Configurations
from dronebuddylib.utils.logging_config import get_logger
from dronebuddylib.utils.utils import config_validity_check

# Get an instance of a logger
logger = get_logger()

"""
This is a wrapper for ttx.
"""

"""
:param engine: The pyttsx engine that was returned by the init_text_to_speech_engine().
:param text: The text to be converted to speechrecognition.

This method will convert the text to speechrecognition and play it.
"""


class TTSTextToSpeechEngineImpl(ISpeechGeneration):
    """
    Initiates the speechrecognition to text engine.
    """

    """
    Required to initialize the pyttsx engine before using the text to voice engine.
    ( since this is the offline model, can only support this voice for the moment)
    :return: The pytts engine.
    """

    def get_class_name(self) -> str:
        return 'TEXT_TO_SPEECH_TTS'

    def get_algorithm_name(self) -> str:
        return 'TTS Text to Speech'

    def __init__(self, engine_configurations: EngineConfigurations):
        """
         Initializes and configures a text-to-speechrecognition engine for generating speechrecognition.

         Args:
             rate (int): The speechrecognition rate in words per minute (default is 150).
             volume (float): The speechrecognition volume level (default is 1.0).
             voice_id (str): The identifier of the desired voice (default is 'TTS_MS_EN-US_ZIRA_11.0').

         Returns:
             pyttsx3.Engine: The initialized text-to-speechrecognition engine instance.

         Example:
             engine = init_text_to_speech_engine(rate=200, volume=0.8, voice_id='TTS_MS_EN-US_DAVID_11.0')
             generate_speech_and_play(engine, "Hello, how can I assist you?")
        Notes:
            since this is the offline model, can only support this voice for the moment
         """
        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())
        configs = engine_configurations.get_configurations_for_engine(self.get_class_name())

        rate = configs.get(Configurations.SPEECH_GENERATION_TTS_RATE, 150)
        volume = configs.get(Configurations.SPEECH_GENERATION_TTS_VOLUME, 1)
        voice_id = configs.get(Configurations.SPEECH_GENERATION_TTS_VOICE_ID, 'TTS_MS_EN-US_ZIRA_11.0')

        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        engine.setProperty("volume", volume)
        engine.setProperty('voice', voice_id)
        logger.debug("Text to speechrecognition: Initialized Text to Speech Engine")
        self.engine = engine

    def change_voice(self, voice_id) -> bool:
        self.engine.setProperty('voice', voice_id)
        return True

    def change_volume(self, volume) -> bool:
        self.engine.setProperty('volume', volume)
        return True

    def change_rate(self, rate) -> bool:
        self.engine.setProperty('rate', rate)
        return True

    def get_current_configs(self) -> dict:
        return {
            Configurations.SPEECH_GENERATION_TTS_RATE: self.engine.getProperty('rate'),
            Configurations.SPEECH_GENERATION_TTS_VOLUME: self.engine.getProperty('volume'),
            Configurations.SPEECH_GENERATION_TTS_VOICE_ID: self.engine.getProperty('voice')
        }

    def get_required_params(self) -> list:
        return []

    def get_optional_params(self) -> list:
        return [Configurations.SPEECH_GENERATION_TTS_RATE, Configurations.SPEECH_GENERATION_TTS_VOLUME,
                Configurations.SPEECH_GENERATION_TTS_VOICE_ID]

    def read_phrase(self, text) -> None:
        """
           Generates speechrecognition from the provided text using a text-to-speechrecognition engine and plays it.

           Args:
               engine (TTS Engine): The text-to-speechrecognition engine instance capable of generating speechrecognition.
               text (str): The text to be converted into speechrecognition and played.

           Returns:
               None

           Example:
               engine = TextToSpeechEngine()
               generate_speech_and_play(engine, "Hello, how can I assist you?")
           """
        logger.info("Text to speechrecognition: " + text)
        self.engine.say(text)
        logger.info("Text to speechrecognition: Done")
        self.engine.runAndWait()
        self.engine.stop()
        return

import pyttsx3

from dronebuddylib.utils.logging_config import get_logger

# Get an instance of a logger
logger = get_logger()
''''This is a wrapper for ttx. '''

'''
:param engine: The pyttsx engine that was returned by the init_text_to_speech_engine().
:param text: The text to be converted to speechRecognition.

this method will convert the text to speechRecognition and play it.
'''


class OffLineTextToSpeechEngine:
    # initiates the speechRecognition to text engine
    '''
    Required to initialize the pyttsx engine before using the text to voice engine.
    :param rate: The rate of the voice. The default is 150.
    :param volume: The volume of the voice. The default is 1.
    :param voice_id: The voice id. The default is 'TTS_MS_EN-US_ZIRA_11.0'.
    ( since this is the offline model, can only support this voice for the moment)
    :return: The pytts engine.

    '''

    def __init__(self, rate=150, volume=1, voice_id='TTS_MS_EN-US_ZIRA_11.0'):
        """
         Initializes and configures a text-to-speechRecognition engine for generating speechRecognition.

         Args:
             rate (int): The speechRecognition rate in words per minute (default is 150).
             volume (float): The speechRecognition volume level (default is 1.0).
             voice_id (str): The identifier of the desired voice (default is 'TTS_MS_EN-US_ZIRA_11.0').

         Returns:
             pyttsx3.Engine: The initialized text-to-speechRecognition engine instance.

         Example:
             engine = init_text_to_speech_engine(rate=200, volume=0.8, voice_id='TTS_MS_EN-US_DAVID_11.0')
             generate_speech_and_play(engine, "Hello, how can I assist you?")
        Notes:
            since this is the offline model, can only support this voice for the moment
         """
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        engine.setProperty("volume", volume)
        engine.setProperty('voice', voice_id)
        logger.debug("Text to speechRecognition: Initialized Text to Speech Engine")
        self.engine = engine

    def generate_speech_and_play(self, text):
        """
           Generates speechRecognition from the provided text using a text-to-speechRecognition engine and plays it.

           Args:
               engine (TTS Engine): The text-to-speechRecognition engine instance capable of generating speechRecognition.
               text (str): The text to be converted into speechRecognition and played.

           Returns:
               None

           Example:
               engine = TextToSpeechEngine()
               generate_speech_and_play(engine, "Hello, how can I assist you?")
           """
        logger.info("Text to speechRecognition: " + text)
        self.engine.say(text)
        logger.info("Text to speechRecognition: Done")
        self.engine.runAndWait()
        self.engine.stop()
        return

    class Voice:

        def __init__(self, r, v):
            logger.info("Text to speechRecognition: Init")
            self.engine = pyttsx3.init()
            self.rate = self.engine.setProperty('rate', r)
            self.volume = self.engine.setProperty("volume", v)

        def get_rate(self):
            return self.rate

        def get_volume(self):
            return self.volume

        def set_rate(self, new_rate):
            self.engine.setProperty('rate', new_rate)
            return

        # This function is to set the volume of the voice. The volume should between 0 and 1.
        def set_volume(self, new_volume):
            self.engine.setProperty('volume', new_volume)
            return

        # This function is to set the texture of the voice, such as language, gender.
        # For more voice_ids, please see the documentation.
        def set_voice_id(self, new_voice_id):
            self.engine.setProperty('voice', new_voice_id)
            return

        # The input is the text. The output is the audio.
        def play_audio(self, text):
            logger.debug("Text to speechRecognition: text to convert :" + text)
            self.engine.say(text)
            logger.debug("Text to speechRecognition: Audio playback completed")
            self.engine.runAndWait()
            self.engine.stop()
            return

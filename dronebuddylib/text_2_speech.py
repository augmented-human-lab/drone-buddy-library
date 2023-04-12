import pyttsx3

from dronebuddylib.logging_config import get_logger

# Get an instance of a logger
logger = get_logger()
''''This is a wrapper for ttx. '''


def generate_speech_and_play(engine, text):
    logger.info("Text to speech: " + text)
    engine.say(text)
    logger.info("Text to speech: Done")
    engine.runAndWait()
    engine.stop()
    return


def init_voice_engine(rate=150, volume=1,voice_id='TTS_MS_EN-US_ZIRA_11.0'):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.setProperty("volume", volume)
    engine.setProperty('voice', voice_id)
    return engine


class Voice:

    def __init__(self, r, v):
        logger.info("Text to speech: Init")
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
        logger.info("Text to speech: " + text)
        self.engine.say(text)
        logger.info("Text to speech: Done")
        self.engine.runAndWait()
        self.engine.stop()
        return

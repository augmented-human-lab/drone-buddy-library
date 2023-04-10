import json

from vosk import Model, KaldiRecognizer

from dronebuddylib.enums import DroneCommands
from dronebuddylib.logging_config import get_logger

# Get an instance of a logger
logger = get_logger()


def init_model(language):
    model = Model(lang=language)
    rec = KaldiRecognizer(model,
                          44100)
    logger.info('Initialized speech recognition model')
    return rec


def recognize_speech(model, audio_feed):
    pcm = audio_feed.read(8192)
    if model.AcceptWaveform(pcm):
        r = model.Result()
        logger.debug('recognized word', r)
        return r
    return None


def recognize_command(model, audio_feed):
    pcm = audio_feed.read(8192)
    if model.AcceptWaveform(pcm):
        r = model.Result()
        return classify(r)


def classify(raw_text):
    raw_text = raw_text.replace('\n', '')
    json_object = json.loads(raw_text)
    text = json_object['text']
    if (text == 'take off') or (text == 'takeoff'):
        return DroneCommands.TAKEOFF
    if text == 'land':
        return DroneCommands.LAND
    if text == 'battery':
        return DroneCommands.BATTERY
    if text == 'up':
        return DroneCommands.UP
    if text == 'down':
        return DroneCommands.DOWN
    return DroneCommands.NONE

import json

import pkg_resources
from vosk import Model, KaldiRecognizer

from dronebuddylib.enums import DroneCommands
from dronebuddylib.logging_config import get_logger

# Get an instance of a logger
logger = get_logger()
queue = []


def init_speech_to_text_engine(language):
    model_path = pkg_resources.resource_filename(__name__, "resources/speechrecognition/vosk-model-small-en-us-0.15")

    model = Model(model_path)
    rec = KaldiRecognizer(model, 44100)
    logger.info('Initialized speech recognition model')
    return rec


def recognize_speech(model, audio_feed, chunk_size=8192):
    pcm = audio_feed.read(chunk_size, exception_on_overflow=False)
    if model.AcceptWaveform(pcm):
        r = model.Result()
        logger.debug('recognized word', r)
        return r
    return None


def recognize_command(model, pcm):
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
    if text == 'stop':
        return DroneCommands.STOP
    if text == 'left':
        return DroneCommands.LEFT
    if text == 'right':
        return DroneCommands.RIGHT
    if text == 'forward':
        return DroneCommands.FORWARD
    if text == 'backward':
        return DroneCommands.BACKWARD
    if text == 'battery':
        return DroneCommands.BATTERY

    return DroneCommands.NONE

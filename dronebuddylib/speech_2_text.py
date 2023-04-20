import json

import pkg_resources
from vosk import Model, KaldiRecognizer

from dronebuddylib.enums import DroneCommands
from dronebuddylib.logging_config import get_logger

# Get an instance of a logger
logger = get_logger()
queue = []

'''
:param language: The language of the model. The default is 'en-US'. (currently only supports this language)
:return: The vosk model.

need to initialize the model before using the speech to text engine.
'''


def init_speech_to_text_engine(language):
    """
     Initializes a speech-to-text engine using the Vosk model for a given language.
     (currently only supports 'en-US' language)

     Args:
     - language: a string representing the language code to use (e.g. 'en-us', 'fr-fr')

     Returns:
     - a Vosk KaldiRecognizer object that can be used for speech recognition
     """

    # Define the path to the Vosk model for the given language
    model_path = pkg_resources.resource_filename(__name__, "resources/speechrecognition/vosk-model-small-en-us-0.15")

    # Load the Vosk model and create a KaldiRecognizer object
    model = Model(model_path)

    # Log that the speech recognition model has been initialized
    vosk_kaldi_model = KaldiRecognizer(model, 44100)
    logger.info('Speech Recognition : Initialized speech recognition model')

    # Return the Vosk KaldiRecognizer object
    return vosk_kaldi_model


def recognize_speech(model, audio_feed):
    """
          Recognizes a text from an audio feed using a given model.

          Args:
          - model: The vosk model that is returned by the init_speech_to_text_engine().
          - audio_feed: a byte string representing the audio feed to recognize, taken by audio_feed.read(num_frames)

          Returns:
          - the text that was recognized, or None if no text was recognized
          """
    if model.AcceptWaveform(audio_feed):
        r = model.Result()
        logger.debug('Speech Recognition : Recognized word : ', r)
        return r
    return None


def recognize_command(model, audio_feed):
    """
       Recognizes a command from an audio feed using a given model.

       Args:
       - model: The vosk model that is returned by the init_speech_to_text_engine().
       - audio_feed: a byte string representing the audio feed to recognize, taken by audio_feed.read(num_frames)

       Returns:
       - a label indicating the recognized command, or None if no command was recognized
       """

    # Check if the model accepts the audio waveform
    if model.AcceptWaveform(audio_feed):
        # Retrieve the result from the model
        r = model.Result()
        logger.debug('Speech Recognition : Recognized command : ', r)

        # Classify the result and return the label
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
    if text == 'speed':
        return DroneCommands.SPEED
    if text == 'height':
        return DroneCommands.HEIGHT
    if text == 'clockwise':
        return DroneCommands.ROTATE_CLOCKWISE
    if text == 'counter clockwise':
        return DroneCommands.ROTATE_COUNTER_CLOCKWISE

    return DroneCommands.NONE

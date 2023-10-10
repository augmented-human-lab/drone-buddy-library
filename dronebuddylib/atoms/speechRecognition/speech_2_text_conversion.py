import json

import pkg_resources
from vosk import Model, KaldiRecognizer

from dronebuddylib.utils.enums import DroneCommands
from dronebuddylib.utils.logging_config import get_logger

# Get an instance of a logger
logger = get_logger()
queue = []

'''
:param language: The language of the model. The default is 'en-US'. (currently only supports this language)
:return: The vosk model.

need to initialize the model before using the speechRecognition to text engine.
'''


def init_speech_to_text_engine(language):
    """
     Initializes a speechRecognition-to-text engine using the Vosk model for a given language.
     (currently only supports 'en-US' language)

     Args:
     - language: a string representing the language code to use (e.g. 'en-us', 'fr-fr')

     Returns:
     - a Vosk KaldiRecognizer object that can be used for speechRecognition recognition
     """

    # Define the path to the Vosk model for the given language
    model_path = pkg_resources.resource_filename(__name__, "resources/speechrecognition/vosk-model-small-en-us-0.15")

    # Load the Vosk model and create a KaldiRecognizer object
    model = Model(model_path)

    # Log that the speechRecognition recognition model has been initialized
    vosk_kaldi_model = KaldiRecognizer(model, 44100)
    logger.info('Speech Recognition : Initialized speechRecognition recognition model')

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
        logger.debug('Speech Recognition : Recognized utterance : ', r)
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
        return r


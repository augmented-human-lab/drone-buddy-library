from vosk import Model, KaldiRecognizer

from dronebuddylib.logging_config import get_logger

# Get an instance of a logger
logger = get_logger()


def init_model():
    model = Model(lang="en-us")
    rec = KaldiRecognizer(model,
                          44100)
    logger.info('Initialized speech recognition model')
    return rec


def recognize(model, audio_feed):
    pcm = audio_feed.read(8192)
    if model.AcceptWaveform(pcm):
        r = model.Result()
        logger.debug('recognized word', r)
        return r
    return None


def spot_words(model, audio_feed, words):
    pcm = audio_feed.read(8192)
    if model.AcceptWaveform(pcm):
        r = model.Result()
        logger.debug('recognized word', r)
        for word in words:
            if word in r:
                logger.debug('matched word', r)
                return r
    return None

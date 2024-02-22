import enum


class ObjectDetectionReturnTypes(enum.Enum):
    """Enum for the return types of the object detection functions."""
    # The object detection function returns a list of objects.
    LABELS = "LABELS"
    # The object detection function returns a dictionary of objects.
    BBOX = "BBOX"

    CONF = "CONF"

    ALL = "ALL"


class Language(enum.Enum):
    ENGLISH = 'en-gb',
    FRENCH = 'FR',


class VisionAlgorithm(enum.Enum):
    YOLO_V8 = 'YOLO_V8',
    GOOGLE_VISION = 'GOOGLE_VISION',


class SpeechGenerationAlgorithm(enum.Enum):
    GOOGLE_TTS_OFFLINE = 'GOOGLE_TTS_OFFLINE',


class SpeechRecognitionAlgorithm(enum.Enum):
    GOOGLE_SPEECH_RECOGNITION = 'GOOGLE_SPEECH_RECOGNITION',
    VOSK_SPEECH_RECOGNITION = 'VOSK_SPEECH_RECOGNITION',
    MULTI_ALGO_SPEECH_RECOGNITION = 'MULTI_ALGO_SPEECH_RECOGNITION',


class SpeechRecognitionMultiAlgoAlgorithmSupportedAlgorithms(enum.Enum):
    GOOGLE = 'GOOGLE',
    IBM = 'IBM',
    VOSK = 'VOSK',
    WIT = 'WIT',
    WHISPER = 'WHISPER',


class IntentRecognitionAlgorithm(enum.Enum):
    CHAT_GPT = 'CHAT_GPT',
    SNIPS_NLU = 'SNIPS_NLU',

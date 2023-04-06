__version__ = "0.1.6"

from .enums import ObjectDetectionReturnTypes
from .object_detection import detect_common_object_labels
from .object_detection import detect_common_objects
from .speech_2_text import init_model
from .speech_2_text import recognize
from .speech_2_text import spot_words
from .utils import get_frames, animate

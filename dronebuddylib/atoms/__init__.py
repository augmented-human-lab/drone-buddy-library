__version__ = "0.2.1"

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from .object_detection_yolo import get_label_yolo
from .object_detection_yolo import init_yolo_engine
from .object_detection_yolo import init_yolo_engine

from .speech_2_text import init_speech_to_text_engine
from .speech_2_text import recognize_command
from .speech_2_text import recognize_speech

from .text_2_speech import generate_speech_and_play
from .text_2_speech import init_text_to_speech_engine

# from .intent_recgnition import init_intent_recognition_engine
# from .intent_recgnition import recognize_intent

from .track import get_bounding_box

from .gesture_recognition import is_pointing
from .gesture_recognition import is_stop_following

from .hand_detection import get_hand_landmark

from .head_bounding import head_bounding_box

from .object_detection import detect_objects



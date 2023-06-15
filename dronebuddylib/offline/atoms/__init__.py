__version__ = "1.0.5"

from .object_detection_yolo import get_label_yolo
from .object_detection_yolo import init_yolo_engine
from .object_detection_yolo import init_yolo_engine

from .speech_2_text import init_speech_to_text_engine
from .speech_2_text import recognize_command
from .speech_2_text import recognize_speech

from .text_2_speech import generate_speech_and_play
from .text_2_speech import init_text_to_speech_engine

from .intent_recgnition import init_intent_recognition_engine
from .intent_recgnition import recognize_intent
from .intent_recgnition import get_intent_name
from .intent_recgnition import get_mentioned_entities
from .intent_recgnition import is_addressed_to_drone
#
from .face_recognition import find_all_the_faces
from .face_recognition import add_people_to_memory

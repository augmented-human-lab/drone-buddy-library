__version__ = "1.0.7"

from .online_speech_2_text_conversion import init_google_speech_engine
from .online_speech_2_text_conversion import recognize_speech

from .online_text_recognition import detect_text
from .online_text_recognition import init_google_vision_engine

from .online_conversation_generation import prompt_chatgpt


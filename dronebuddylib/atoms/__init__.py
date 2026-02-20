__version__ = "2.0.33"

# from .facerecognition.face_recognition_engine import FaceRecognitionEngine

# from .intentrecognition.intent_recognition_engine import IntentRecognitionEngine

# from .objectdetection.object_detection_engine import ObjectDetectionEngine

# from .speechgeneration.speech_generation_engine import SpeechGenerationEngine

# from .speechrecognition.speech_recognition_engine import SpeechRecognitionEngine

# from .textrecognition.text_recognition_engine import TextRecognitionEngine

# from .facerecognition.face_recognition_engine import FaceRecognitionEngine

from .navigation.navigation_engine import NavigationEngine
from .navigation.tello_waypoint_nav_utils.tello_waypoint_nav_coordinator import NavigationInstruction

# Planning module - VLM-based planner for intelligent drone task planning
from .planning.planner_engine import PlannerEngine
from .planning.planner_executor import PlannerExecutor
from .planning.planner_agent import PlannerAgent
from .planning.planner_models import (
    PlannerAction,
    PlannerActionType,
    ActionPlan,
    PlannerState,
    PlannerSessionResult
)
# ScanResult and detection types are now in the navigation module's TelloNavExtra
from .navigation.tello_waypoint_nav_utils.tello_nav_extra import (
    ScanResult,
    FrameDetection,
    DetectionResult,
    TelloNavExtra
)

# from .bodyfeatureextraction.hand_feature_extraction_impl import HandFeatureExtractionImpl
# from .bodyfeatureextraction.body_feature_extraction_impl import BodyFeatureExtractionImpl
# from .bodyfeatureextraction.head_feature_extraction_impl import HeadFeatureExtractionImpl

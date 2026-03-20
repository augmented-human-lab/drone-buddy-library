"""Planning API exports.

This package wires together:
- plan generation via VLM providers,
- execution via the navigation engine,
- optional GUI/session tooling.
"""

from dronebuddylib.atoms.planning.planner_engine import PlannerEngine
from dronebuddylib.atoms.planning.planner_agent import PlannerAgent
from dronebuddylib.atoms.planning.planner_executor import PlannerExecutor
from dronebuddylib.atoms.planning.planner_configs import PlannerConfigs
from dronebuddylib.atoms.planning.planner_models import (
    PlannerAction,
    PlannerActionType,
    ActionPlan,
    PlannerState,
    PlannerSessionResult
)

# Import ScanResult and detection types from navigation module (where they now live)
from dronebuddylib.atoms.navigation.tello_waypoint_nav_utils.tello_nav_extra import (
    ScanResult,
    FrameDetection,
    DetectionResult
)

# Import VLM client for multi-provider support
from dronebuddylib.atoms.planning.vlm_client import (
    VLMProvider,
    VLMMessage,
    VLMResponse,
    BaseVLMClient,
    OpenAIClient,
    AnthropicClient,
    GoogleClient,
    create_vlm_client
)

# Import GUI components
from dronebuddylib.atoms.planning.planner_gui import (
    PlannerGUI,
    PlannerGUIApp,
    MessageType
)

__all__ = [
    # Engine and agents
    'PlannerEngine',
    'PlannerAgent',
    'PlannerExecutor',
    'PlannerConfigs',
    # Models
    'PlannerAction',
    'PlannerActionType',
    'ActionPlan',
    'PlannerState',
    'PlannerSessionResult',
    # Re-exported from navigation for convenience
    'ScanResult',
    'FrameDetection',
    'DetectionResult',
    # VLM Client (multi-provider support)
    'VLMProvider',
    'VLMMessage',
    'VLMResponse',
    'BaseVLMClient',
    'OpenAIClient',
    'AnthropicClient',
    'GoogleClient',
    'create_vlm_client',
    # GUI components
    'PlannerGUI',
    'PlannerGUIApp',
    'MessageType',
]

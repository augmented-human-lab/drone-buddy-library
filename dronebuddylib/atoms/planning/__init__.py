"""
VLM-based Planning Module for Drone Buddy Library.

This module provides a high-level planner that uses a Vision-Language Model (VLM) 
to plan and execute sequences of actions for the drone to find specific items.

Supports multiple VLM providers:
- OpenAI (GPT-4, GPT-4o, GPT-5)
- Anthropic (Claude-3.5-Sonnet, Claude-3-Opus)
- Google (Gemini-1.5-Pro, Gemini-1.5-Flash)

Components:
- PlannerEngine: High-level API for VLM-based object finding
- PlannerAgent: VLM-based agent for generating action plans  
- PlannerExecutor: Control flow executor for executing planned actions
- VLM Client: Provider-agnostic abstraction for different VLM providers

The YOLO detection is integrated into the navigation module's TelloNavExtra class
via the scan_with_detection() method.

Example Usage (EngineConfigurations - Standard Pattern):
    from dronebuddylib import EngineConfigurations, AtomicEngineConfigurations
    from dronebuddylib.atoms.planning import PlannerEngine
    
    # Configure like other atoms in the library
    config = EngineConfigurations({})
    config.add_configuration(AtomicEngineConfigurations.PLANNER_VLM_PROVIDER, "openai")
    config.add_configuration(AtomicEngineConfigurations.PLANNER_VLM_API_KEY, "your-key")
    config.add_configuration(AtomicEngineConfigurations.PLANNER_VLM_MODEL, "gpt-4o")
    config.add_configuration(AtomicEngineConfigurations.PLANNER_YOLO_ONNX_MODEL_PATH, "models/yolo.onnx")
    
    engine = PlannerEngine(config)
    result = engine.find_object("Find my coffee cup")
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

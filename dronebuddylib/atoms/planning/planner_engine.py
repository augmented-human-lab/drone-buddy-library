"""
Planner Engine - High-level API for VLM-based drone task planning.

This module provides the PlannerEngine class which serves as the main entry point
for the VLM-based planning functionality, similar to how NavigationEngine provides
the interface for navigation.

Supports multiple VLM providers:
- OpenAI (GPT-4, GPT-4o, GPT-5)
- Anthropic (Claude-3.5-Sonnet, Claude-3-Opus)
- Google (Gemini-1.5-Pro, Gemini-1.5-Flash)
"""

from typing import Optional, Callable, List, Union

from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import AtomicEngineConfigurations
from dronebuddylib.atoms.planning.planner_executor import PlannerExecutor
from dronebuddylib.atoms.planning.planner_configs import PlannerConfigs
from dronebuddylib.atoms.planning.planner_models import (
    PlannerSessionResult,
    ActionPlan,
    PlannerState
)
from dronebuddylib.utils.logger import Logger

logger = Logger()


class PlannerEngine:
    """
    High-level engine for VLM-based drone task planning.
    
    This engine provides a simplified interface for using the VLM planner to
    find objects using a Tello drone. It wraps the PlannerExecutor and provides
    convenient methods for common operations.
    
    Supports multiple VLM providers (OpenAI, Anthropic, Google).
    
    Detection System:
        The planner uses a dual detection system based on VLM classification:
        - COCO class objects: Uses standard YOLO ONNX model (fast, 80 classes)
        - Non-COCO objects: Uses YOLO-World PyTorch model (open-vocabulary)
    
    Obstacle Avoidance:
        The planner supports MiDaS depth-based obstacle detection during navigation.
        When enabled, the drone will pause before forward movements if obstacles
        are detected in the path, and wait until the path is clear.
    
    Creating an Engine:
        # Recommended: Use from_config() with PlannerConfigs
        config = PlannerConfigs.from_json_file("config.json")
        engine = PlannerEngine.from_config(config)
        
        # Alternative: Use EngineConfigurations
        eng_config = EngineConfigurations({})
        eng_config.add_configuration(AtomicEngineConfigurations.PLANNER_VLM_PROVIDER, "openai")
        eng_config.add_configuration(AtomicEngineConfigurations.PLANNER_VLM_API_KEY, "your-key")
        engine = PlannerEngine(eng_config)
    
    Configuration Options (via EngineConfigurations):
        - PLANNER_VLM_PROVIDER: VLM provider (openai, anthropic, google) (default: openai)
        - PLANNER_VLM_API_KEY: API key for VLM provider (required)
        - PLANNER_VLM_MODEL: VLM model name (provider-specific default)
        - PLANNER_YOLO_ONNX_MODEL_PATH: Path to ONNX YOLO model (required)
        - PLANNER_YOLO_CONF_THRESHOLD: Detection confidence threshold (default: 0.3)
        - PLANNER_YOLO_IOU_THRESHOLD: NMS IOU threshold (default: 0.45)
        - PLANNER_YOLO_WORLD_MODEL_PATH: Path to YOLO-World model (for non-COCO objects)
        - PLANNER_YOLO_WORLD_CONF_THRESHOLD: YOLO-World confidence (default: 0.025)
        - PLANNER_MAX_REPLAN_ATTEMPTS: Max re-planning attempts (default: 2)
        - NAVIGATION_TELLO_WAYPOINT_FILE: Waypoint file to use
        - NAVIGATION_TELLO_WAYPOINT_DIR: Waypoint directory
        - NAVIGATION_TELLO_IMAGE_DIR: Directory for scan images
        - NAVIGATION_TELLO_WAYPOINT_MIDAS_MODEL_PATH: Path to MiDaS ONNX model
        - NAVIGATION_TELLO_WAYPOINT_OBSTACLE_DETECTION_MODE: ObstacleDetectionMode enum
        - NAVIGATION_TELLO_WAYPOINT_TAKEOFF_ALTITUDE_CM: Target altitude in cm after takeoff (0 = no adjustment)
    
    Example:
        from dronebuddylib.models.enums import ObstacleDetectionMode
        
        config = PlannerConfigs(
            vlm_provider="anthropic",
            vlm_api_key="your-anthropic-key",
            vlm_model="claude-3-5-sonnet-20241022",
            yolo_model_path="models/yolov8n.onnx",
            yolo_world_model_path="models/yolov8m-worldv2.pt",
            midas_model_path="models/midas_v21_small_256.onnx",
            obstacle_detection_mode="MEDIUM"
        )
        engine = PlannerEngine.from_config(config)
        result = engine.find_object("Find my coffee cup")
    """
    
    @classmethod
    def from_config(
        cls, 
        config: PlannerConfigs,
        user_input_callback: Optional[Callable[[str], str]] = None
    ) -> 'PlannerEngine':
        """
        Create a PlannerEngine from a PlannerConfigs object.
        
        This is the recommended way to create a PlannerEngine as it supports
        the full range of configuration options including multi-provider VLM.
        
        Args:
            config: PlannerConfigs with all settings
            user_input_callback: Optional callback for user input
            
        Returns:
            Configured PlannerEngine instance
            
        Example:
            config = PlannerConfigs.from_json_file("planner_config.json")
            engine = PlannerEngine.from_config(config)
        """
        # Create an EngineConfigurations from PlannerConfigs
        eng_config = EngineConfigurations({})
        eng_config.add_configuration(
            AtomicEngineConfigurations.PLANNER_VLM_PROVIDER, 
            config.vlm_provider
        )
        eng_config.add_configuration(
            AtomicEngineConfigurations.PLANNER_VLM_API_KEY, 
            config.vlm_api_key
        )
        eng_config.add_configuration(
            AtomicEngineConfigurations.PLANNER_VLM_MODEL, 
            config.vlm_model
        )
        eng_config.add_configuration(
            AtomicEngineConfigurations.PLANNER_YOLO_ONNX_MODEL_PATH, 
            config.yolo_model_path
        )
        if config.waypoint_file_path:
            eng_config.add_configuration(
                AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_FILE, 
                config.waypoint_file_path
            )
        
        # Create instance using the converted config
        instance = cls.__new__(cls)
        instance._init_from_planner_config(config, user_input_callback)
        return instance
    
    def _init_from_planner_config(
        self,
        config: PlannerConfigs,
        user_input_callback: Optional[Callable[[str], str]] = None
    ):
        """Internal initialization from PlannerConfigs."""
        logger.log_info('PlannerEngine', 'Initializing Planner Engine from PlannerConfigs...')
        
        self.api_key = config.vlm_api_key
        self.yolo_model_path = config.yolo_model_path
        self.vlm_model = config.vlm_model
        self.vlm_provider = config.vlm_provider
        self.waypoint_file = config.waypoint_file_path
        self.waypoint_dir = None
        self.image_dir = config.scan_image_directory
        self.nav_config = None
        self.user_input_callback = user_input_callback
        
        # Create executor from config
        self.executor = PlannerExecutor.from_config(config)
        
        logger.log_success('PlannerEngine', 'Planner Engine initialized successfully')
    
    def __init__(
        self, 
        config: EngineConfigurations,
        user_input_callback: Optional[Callable[[str], str]] = None
    ):
        """
        Initialize the Planner Engine from EngineConfigurations.
        
        For multi-provider support, prefer using from_config() with PlannerConfigs.
        
        Args:
            config: Engine configuration with required settings
            user_input_callback: Optional callback for user input (uses input() if None)
            
        Raises:
            ValueError: If required configuration is missing
        """
        logger.log_info('PlannerEngine', 'Initializing Planner Engine...')
        
        # Get all configurations (need both PLANNER_* and NAVIGATION_* configs)
        configs = config.get_configurations()
        
        # Required configurations
        self.api_key = configs.get(AtomicEngineConfigurations.PLANNER_VLM_API_KEY)
        self.yolo_model_path = configs.get(AtomicEngineConfigurations.PLANNER_YOLO_ONNX_MODEL_PATH)
        
        if not self.api_key:
            raise ValueError("PLANNER_VLM_API_KEY is required")
        if not self.yolo_model_path:
            raise ValueError("PLANNER_YOLO_ONNX_MODEL_PATH is required")
        
        # Required: Waypoint file for navigation
        self.waypoint_file = configs.get(AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_FILE)
        if not self.waypoint_file:
            raise ValueError("NAVIGATION_TELLO_WAYPOINT_FILE is required for planner navigation")
        
        # Provider configuration (new for multi-provider support)
        self.vlm_provider = configs.get(
            AtomicEngineConfigurations.PLANNER_VLM_PROVIDER,
            "openai"
        )
        
        # Optional configurations
        self.vlm_model = configs.get(
            AtomicEngineConfigurations.PLANNER_VLM_MODEL, 
            None  # Let the executor choose default based on provider
        )
        self.vlm_temperature = configs.get(
            AtomicEngineConfigurations.PLANNER_VLM_TEMPERATURE,
            0.3  # Default temperature for more deterministic planning
        )
        self.waypoint_dir = configs.get(
            AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_DIR
        )
        self.image_dir = configs.get(
            AtomicEngineConfigurations.NAVIGATION_TELLO_IMAGE_DIR
        )
        self.max_replan_attempts = configs.get(
            AtomicEngineConfigurations.PLANNER_MAX_REPLAN_ATTEMPTS,
            2  # Default max replan attempts
        )
        self.yolo_conf_threshold = configs.get(
            AtomicEngineConfigurations.PLANNER_YOLO_CONF_THRESHOLD,
            0.25  # Default confidence threshold
        )
        self.yolo_iou_threshold = configs.get(
            AtomicEngineConfigurations.PLANNER_YOLO_IOU_THRESHOLD,
            0.45  # Default IOU threshold
        )
        
        # YOLO-World configuration for open-vocabulary detection
        self.yolo_world_model_path = configs.get(
            AtomicEngineConfigurations.PLANNER_YOLO_WORLD_MODEL_PATH,
            ""  # Optional - only needed for non-COCO class detection
        )
        self.yolo_world_conf_threshold = configs.get(
            AtomicEngineConfigurations.PLANNER_YOLO_WORLD_CONF_THRESHOLD,
            0.025  # Default lower threshold for YOLO-World
        )
        
        # MiDaS obstacle detection configuration
        self.midas_model_path = configs.get(
            AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_MIDAS_MODEL_PATH,
            ""  # Optional - obstacle detection disabled if not provided
        )
        obstacle_mode = configs.get(
            AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_OBSTACLE_DETECTION_MODE,
            None
        )
        # Convert ObstacleDetectionMode enum to string if needed
        if obstacle_mode is not None:
            from dronebuddylib.models.enums import ObstacleDetectionMode
            if isinstance(obstacle_mode, ObstacleDetectionMode):
                self.obstacle_detection_mode = obstacle_mode.name
            else:
                self.obstacle_detection_mode = str(obstacle_mode)
        else:
            self.obstacle_detection_mode = "OFF"
        
        # Takeoff altitude configuration
        self.takeoff_altitude_cm = configs.get(
            AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_TAKEOFF_ALTITUDE_CM,
            0  # 0 = no adjustment, drone stays at default ~80 cm
        )
        
        # Mission pad alignment configuration
        self.mission_pad_enabled = configs.get(
            AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_MISSION_PAD_ENABLED,
            False
        )
        
        # Store config for navigation
        self.nav_config = config
        self.user_input_callback = user_input_callback
        
        # Create executor with multi-provider support
        self.executor = PlannerExecutor(
            provider=self.vlm_provider,
            api_key=self.api_key,
            yolo_model_path=self.yolo_model_path,
            yolo_conf_threshold=self.yolo_conf_threshold,
            yolo_iou_threshold=self.yolo_iou_threshold,
            yolo_world_model_path=self.yolo_world_model_path,
            yolo_world_conf_threshold=self.yolo_world_conf_threshold,
            midas_model_path=self.midas_model_path,
            obstacle_detection_mode=self.obstacle_detection_mode,
            waypoint_file=self.waypoint_file,
            waypoint_dir=self.waypoint_dir,
            image_dir=self.image_dir,
            model=self.vlm_model,
            temperature=self.vlm_temperature,
            max_replan_attempts=self.max_replan_attempts,
            takeoff_altitude_cm=self.takeoff_altitude_cm,
            mission_pad_enabled=self.mission_pad_enabled,
            nav_config=self.nav_config,
            user_input_callback=self.user_input_callback
        )
        
        logger.log_success('PlannerEngine', 'Planner Engine initialized successfully')
    
    def get_class_name(self) -> str:
        """Get the class name for configuration lookup."""
        return 'PLANNER_ENGINE'
    
    def find_object(self, request: str) -> PlannerSessionResult:
        """
        Execute a complete object search session.
        
        This is the main method for finding objects. It takes a natural language
        request, generates an action plan using the VLM, and executes it using
        the drone.
        
        Args:
            request: Natural language request (e.g., "Find my red coffee cup")
            
        Returns:
            PlannerSessionResult with session outcome and details
            
        Example:
            result = engine.find_object("Find my laptop")
            if result.success:
                print(f"Found at {result.found_at_waypoint}")
        """
        logger.log_info('PlannerEngine', f'Starting object search: "{request}"')
        return self.executor.execute(request)
    
    def generate_plan_only(
        self, 
        request: str, 
        waypoint_names: List[str]
    ) -> Optional[ActionPlan]:
        """
        Generate an action plan without executing it.
        
        Useful for previewing what the VLM would plan before running.
        
        Args:
            request: Natural language request
            waypoint_names: List of available waypoint names
            
        Returns:
            ActionPlan object or None if generation failed
        """
        logger.log_info('PlannerEngine', f'Generating plan for: "{request}"')
        return self.executor.planner_agent.generate_plan(
            user_request=request,
            waypoint_names=waypoint_names,
            current_waypoint="START"
        )
    
    def get_state(self) -> PlannerState:
        """Get the current state of the planner executor."""
        return self.executor.state
    
    def get_required_params(self) -> list:
        """Get list of required configuration parameters."""
        return [
            AtomicEngineConfigurations.PLANNER_VLM_API_KEY,
            AtomicEngineConfigurations.PLANNER_YOLO_ONNX_MODEL_PATH
        ]
    
    def get_optional_params(self) -> list:
        """Get list of optional configuration parameters."""
        return [
            AtomicEngineConfigurations.PLANNER_VLM_MODEL,
            AtomicEngineConfigurations.PLANNER_YOLO_CONF_THRESHOLD,
            AtomicEngineConfigurations.PLANNER_YOLO_IOU_THRESHOLD,
            AtomicEngineConfigurations.PLANNER_MAX_REPLAN_ATTEMPTS,
            AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_FILE,
            AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_DIR,
            AtomicEngineConfigurations.NAVIGATION_TELLO_IMAGE_DIR,
            AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_TAKEOFF_ALTITUDE_CM,
            AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_MISSION_PAD_ENABLED
        ]

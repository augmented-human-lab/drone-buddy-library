"""
Planner Executor - Control flow executor for VLM-planned drone actions.

This module provides the PlannerExecutor class that orchestrates the execution
of VLM-generated action plans, handling the complete workflow from user request
to object finding, confirmation, and session completion.

This executor uses:
- PlannerAgent: For VLM-based plan generation
- NavigationEngine: For drone navigation AND 360° scan with YOLO detection
  (unified interface via scan_with_detection wrapper)
"""

import os
import time
import json
from datetime import datetime
from typing import List, Optional, Callable, Dict, Any

from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import AtomicEngineConfigurations, NavigationAlgorithm
from dronebuddylib.atoms.navigation.navigation_engine import NavigationEngine
from dronebuddylib.atoms.navigation.tello_waypoint_nav_utils.tello_waypoint_nav_coordinator import (
    TelloWaypointNavCoordinator, 
    NavigationInstruction
)
from dronebuddylib.atoms.navigation.tello_waypoint_nav_utils.tello_nav_extra import ScanResult
from dronebuddylib.atoms.planning.planner_agent import PlannerAgent
from dronebuddylib.atoms.planning.planner_models import (
    ActionPlan,
    PlannerAction,
    PlannerActionType,
    PlannerState,
    PlannerSessionResult
)
from dronebuddylib.atoms.planning.planner_configs import PlannerConfigs
from dronebuddylib.utils.logger import Logger

logger = Logger()

# YOLO COCO class names (80 classes) - for validation
YOLO_COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class PlannerExecutor:
    """
    Control flow executor for VLM-planned drone object search operations.
    
    This class manages the complete execution pipeline:
    1. Accept user request and generate action plan via VLM
    2. Execute actions sequentially (navigate, scan)
    3. Detect target objects using YOLO
    4. Handle user confirmation flow
    5. Support re-planning after failed searches
    6. Manage drone state and ensure safe return
    
    Supports multiple VLM providers through PlannerConfigs:
    - OpenAI (GPT-4, GPT-4o, GPT-5)
    - Anthropic (Claude)
    - Google (Gemini)
    
    The executor implements a state machine to track the session progress
    and handle various scenarios (object found, not found, user confirmation, etc.)
    
    Example:
        # Option 1: Using PlannerConfigs (recommended)
        config = PlannerConfigs(
            vlm_provider="openai",
            vlm_api_key="your-key",
            yolo_model_path="yolov8n_640x640.onnx",
            waypoint_file_path="my_waypoints.json"
        )
        executor = PlannerExecutor.from_config(config)
        
        # Option 2: Direct initialization
        executor = PlannerExecutor(
            provider="openai",
            api_key="your-key",
            yolo_model_path="yolov8n_640x640.onnx"
        )
        result = executor.execute("Find my coffee cup")
    """
    
    # Maximum number of re-planning attempts
    MAX_REPLAN_ATTEMPTS = 2
    
    def __init__(
        self,
        provider: str = "openai",
        api_key: str = "",
        yolo_model_path: str = "",
        yolo_conf_threshold: float = 0.25,
        yolo_iou_threshold: float = 0.45,
        yolo_world_model_path: str = "",
        yolo_world_conf_threshold: float = 0.025,
        midas_model_path: str = "",
        obstacle_detection_mode: str = "OFF",
        waypoint_file: Optional[str] = None,
        waypoint_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_replan_attempts: int = 2,
        nav_config: Optional[EngineConfigurations] = None,
        user_input_callback: Optional[Callable[[str], str]] = None
    ):
        """
        Initialize the Planner Executor.
        
        Args:
            provider: VLM provider ("openai", "anthropic", "google")
            api_key: API key for the VLM provider
            yolo_model_path: Path to ONNX YOLO model file (for COCO 80-class detection)
            yolo_conf_threshold: YOLO confidence threshold (default: 0.25)
            yolo_iou_threshold: YOLO IOU threshold for NMS (default: 0.45)
            yolo_world_model_path: Path to YOLO-World PyTorch model (for open-vocabulary detection)
            yolo_world_conf_threshold: YOLO-World confidence threshold (default: 0.025)
            midas_model_path: Path to MiDaS ONNX model for depth-based obstacle detection
            obstacle_detection_mode: Obstacle detection sensitivity (OFF, LOW, MEDIUM, HIGH, VERY_HIGH)
            waypoint_file: Specific waypoint file to use (optional)
            waypoint_dir: Directory containing waypoint files (optional)
            image_dir: Directory for saving scan images (optional)
            model: VLM model name (uses provider default if not specified)
            temperature: VLM temperature (0.0-1.0)
            max_replan_attempts: Maximum replanning attempts
            nav_config: Navigation engine configuration (optional)
            user_input_callback: Callback function for getting user input
                                 If None, uses input() for console interaction
        """
        logger.log_info('PlannerExecutor', 'Initializing Planner Executor...')
        
        # Store configuration
        self.provider = provider
        self.api_key = api_key
        self.yolo_model_path = yolo_model_path
        self.yolo_conf_threshold = yolo_conf_threshold
        self.yolo_iou_threshold = yolo_iou_threshold
        self.yolo_world_model_path = yolo_world_model_path
        self.yolo_world_conf_threshold = yolo_world_conf_threshold
        self.midas_model_path = midas_model_path
        self.obstacle_detection_mode = obstacle_detection_mode
        self.waypoint_file = waypoint_file
        self.waypoint_dir = waypoint_dir
        self.image_dir = image_dir
        self.model = model
        self.MAX_REPLAN_ATTEMPTS = max_replan_attempts
        
        # User interaction callback
        self.user_input_callback = user_input_callback or self._default_user_input
        
        # Initialize VLM Planner Agent (multi-provider)
        logger.log_debug('PlannerExecutor', f'Initializing Planner Agent with provider: {provider}...')
        self.planner_agent = PlannerAgent(
            provider=provider,
            api_key=api_key,
            model=model,
            temperature=temperature
        )
        
        # Initialize navigation engine
        logger.log_debug('PlannerExecutor', 'Initializing Navigation Engine...')
        if nav_config is None:
            nav_config = EngineConfigurations({})
        if waypoint_file:
            nav_config.add_configuration(
                AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_FILE, 
                waypoint_file
            )
        if waypoint_dir:
            nav_config.add_configuration(
                AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_DIR,
                waypoint_dir
            )
        if image_dir:
            nav_config.add_configuration(
                AtomicEngineConfigurations.NAVIGATION_TELLO_IMAGE_DIR,
                image_dir
            )
        
        # Add MiDaS obstacle detection configuration
        if midas_model_path:
            nav_config.add_configuration(
                AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_MIDAS_MODEL_PATH,
                midas_model_path
            )
        if obstacle_detection_mode and obstacle_detection_mode != "OFF":
            # Import ObstacleDetectionMode here to convert string to enum
            from dronebuddylib.models.enums import ObstacleDetectionMode
            try:
                mode = ObstacleDetectionMode[obstacle_detection_mode.upper()]
                nav_config.add_configuration(
                    AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_OBSTACLE_DETECTION_MODE,
                    mode
                )
                logger.log_info('PlannerExecutor', f'Obstacle detection enabled: {obstacle_detection_mode}')
            except KeyError:
                logger.log_warning('PlannerExecutor', f'Invalid obstacle detection mode: {obstacle_detection_mode}')
        
        self.nav_config = nav_config
        self.nav_engine: Optional[NavigationEngine] = None
        
        # State tracking
        self.state = PlannerState.IDLE
        self.current_plan: Optional[ActionPlan] = None
        self.target_object: str = ""
        self.current_waypoint: str = "START"
        self.waypoints_visited: List[str] = []
        self.scans_performed: int = 0
        self.replan_count: int = 0
        self.session_start_time: float = 0
        
        # Available waypoints (loaded from file)
        self.available_waypoints: List[str] = []
        # Mapping from waypoint ID to English name (for user-friendly display)
        self.waypoint_id_to_name: Dict[str, str] = {}
        self.waypoint_name_to_id: Dict[str, str] = {}
        
        logger.log_success('PlannerExecutor', 'Planner Executor initialized successfully')
    
    @classmethod
    def from_config(cls, config: PlannerConfigs, 
                    user_input_callback: Optional[Callable[[str], str]] = None) -> 'PlannerExecutor':
        """
        Create a PlannerExecutor from a PlannerConfigs object.
        
        This is the recommended way to create a PlannerExecutor as it uses
        all settings from the configuration file.
        
        Args:
            config: PlannerConfigs instance with all settings
            user_input_callback: Optional callback for user input
            
        Returns:
            Configured PlannerExecutor instance
        """
        return cls(
            provider=config.vlm_provider,
            api_key=config.vlm_api_key,
            yolo_model_path=config.yolo_model_path,
            yolo_conf_threshold=config.yolo_confidence_threshold,
            yolo_iou_threshold=config.yolo_iou_threshold,
            yolo_world_model_path=config.yolo_world_model_path,
            yolo_world_conf_threshold=config.yolo_world_confidence_threshold,
            midas_model_path=config.midas_model_path,
            obstacle_detection_mode=config.obstacle_detection_mode,
            waypoint_file=config.waypoint_file_path,
            waypoint_dir=config.waypoint_directory,
            image_dir=config.scan_image_directory,
            model=config.vlm_model,
            temperature=config.vlm_temperature,
            max_replan_attempts=config.max_replan_attempts,
            user_input_callback=user_input_callback
        )
    
    def _get_waypoint_display_name(self, waypoint_id: str) -> str:
        """
        Get the user-friendly display name for a waypoint.
        
        Converts waypoint IDs (like SWP_002_IWP_001) to English names (like "kitchen").
        
        Args:
            waypoint_id: The waypoint ID or name
            
        Returns:
            The user-friendly English name, or the original ID if no mapping exists
        """
        if not waypoint_id:
            return waypoint_id
        # Try direct lookup
        if waypoint_id in self.waypoint_id_to_name:
            return self.waypoint_id_to_name[waypoint_id]
        # Return original if no mapping found
        return waypoint_id
    
    def _default_user_input(self, prompt: str) -> str:
        """Default user input using console."""
        return input(prompt)
    
    def _send_message_to_user(self, message: str):
        """Send a message to the user (print to console)."""
        print(f"\n Drone Assistant: {message}\n")
    
    def execute(self, user_request: str) -> PlannerSessionResult:
        """
        Execute a complete object search session based on user request.
        
        This is the main entry point for running the planner. It handles the
        complete workflow from request to completion.
        
        Args:
            user_request: Natural language request (e.g., "Find my coffee cup")
            
        Returns:
            PlannerSessionResult with session outcome and details
        """
        self.session_start_time = time.time()
        self.state = PlannerState.PLANNING
        self.waypoints_visited = []
        self.scans_performed = 0
        self.replan_count = 0
        
        # Reset session termination flags for new session
        TelloWaypointNavCoordinator._session_terminated = False
        TelloWaypointNavCoordinator._obstacle_timeout_occurred = False
        
        logger.log_info('PlannerExecutor', f'Starting session for request: "{user_request}"')
        self._send_message_to_user(f'Received request: "{user_request}"')
        self._send_message_to_user('Let me plan the best route to search for this item...')
        
        try:
            # Initialize navigation engine and get waypoints
            if not self._initialize_navigation():
                return self._create_error_result("Failed to initialize navigation")
            
            # Load available waypoints
            if not self._load_waypoints():
                return self._create_error_result("Failed to load waypoints")
            
            self._send_message_to_user(f'I have access to {len(self.available_waypoints)} locations: {", ".join(self.available_waypoints)}')
            
            # Generate initial plan
            self.current_plan = self.planner_agent.generate_plan(
                user_request=user_request,
                waypoint_names=self.available_waypoints,
                current_waypoint=self.current_waypoint
            )
            
            if not self.current_plan:
                return self._create_error_result("Failed to generate action plan")
            
            self.target_object = self.current_plan.target_object
            
            # Log detection mode based on VLM classification
            if self.current_plan.is_coco_class:
                logger.log_info('PlannerExecutor', 
                    f'Target object (COCO class): {self.target_object} - using YOLO ONNX')
            else:
                detection_targets = self.current_plan.get_detection_targets()
                logger.log_info('PlannerExecutor', 
                    f'Target object (non-COCO): {self.target_object} - using YOLO-World')
                logger.log_debug('PlannerExecutor', 
                    f'YOLO-World detection targets: {detection_targets}')
            
            # Show user both the object and detection mode
            if self.current_plan.user_description:
                mode_str = "standard YOLO" if self.current_plan.is_coco_class else "YOLO-World open-vocabulary"
                self._send_message_to_user(
                    f'I will search for the item you described, detected as "{self.target_object}", using {mode_str}.'
                )
            else:
                self._send_message_to_user(f'I will search for: "{self.target_object}"')
            
            # Show related objects for YOLO-World detection
            if not self.current_plan.is_coco_class and self.current_plan.related_objects:
                self._send_message_to_user(
                    f'I will also look for related items: {", ".join(self.current_plan.related_objects)}'
                )
            
            # Format the plan reasoning with each numbered point on a new line
            plan_reasoning = self.current_plan.reasoning
            # Split numbered points onto separate lines for readability
            import re
            formatted_reasoning = re.sub(r'\s*\((\d+)\)\s*', r'\n(\1) ', plan_reasoning).strip()
            self._send_message_to_user(f'Plan:\n{formatted_reasoning}')
            
            # Execute the plan
            result = self._execute_plan()
            
            return result
            
        except Exception as e:
            logger.log_error('PlannerExecutor', f'Session error: {e}')
            self._safe_return_to_start()
            return self._create_error_result(str(e))
    
    def _initialize_navigation(self) -> bool:
        """Initialize the navigation engine."""
        try:
            # Set up external callback on coordinator class before creating engine
            # This callback is called when nav_manager is created in run_goto_mode
            if hasattr(self, '_video_source_callback') and self._video_source_callback:
                TelloWaypointNavCoordinator._external_nav_manager_callback = self._video_source_callback
            
            self.nav_engine = NavigationEngine(
                NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT,
                self.nav_config
            )
            
            return True
        except Exception as e:
            logger.log_error('PlannerExecutor', f'Navigation init failed: {e}')
            return False
    
    def set_video_source_callback(self, callback):
        """
        Set a callback to receive nav_manager when navigation initializes.
        
        The callback will receive (nav_manager) when the navigation system
        creates the WaypointNavigationManager. The GUI can then register
        its frame callback with the nav_manager.
        
        Args:
            callback: Function(nav_manager) to call
        """
        self._video_source_callback = callback
    
    def _load_waypoints(self) -> bool:
        """
        Load available waypoints from the 2D waypoint file.
        
        Loads ALL waypoint names including Super Waypoints and Inner Waypoints.
        Also builds ID-to-name mapping for user-friendly display.
        """
        try:
            # Reset mappings
            self.waypoint_id_to_name = {}
            self.waypoint_name_to_id = {}
            
            # Get the coordinator instance to access waypoint data
            coordinator = TelloWaypointNavCoordinator._active_instance
            
            if coordinator and hasattr(coordinator, 'nav_manager'):
                nav_manager = coordinator.nav_manager
                # Use all_waypoints (contains both Super and Inner waypoints)
                if hasattr(nav_manager, 'all_waypoints') and nav_manager.all_waypoints:
                    self.available_waypoints = []
                    for wp_id, wp in nav_manager.all_waypoints.items():
                        self.available_waypoints.append(wp.name)
                        self.waypoint_id_to_name[wp_id] = wp.name
                        self.waypoint_id_to_name[wp.name] = wp.name  # Also map name to itself
                        self.waypoint_name_to_id[wp.name.lower()] = wp_id
                else:
                    self.available_waypoints = []
            else:
                # Fallback: load from file directly
                waypoint_file_path = self._find_waypoint_file()
                if waypoint_file_path:
                    with open(waypoint_file_path, 'r') as f:
                        data = json.load(f)
                    
                    self.available_waypoints = []
                    
                    # 2D format (super_waypoints)
                    if 'super_waypoints' in data:
                        for swp in data['super_waypoints']:
                            # Add Super Waypoint
                            swp_id = swp.get('id', '')
                            swp_name = swp.get('name', swp_id)
                            if swp_name:
                                self.available_waypoints.append(swp_name)
                                self.waypoint_id_to_name[swp_id] = swp_name
                                self.waypoint_id_to_name[swp_name] = swp_name
                                self.waypoint_name_to_id[swp_name.lower()] = swp_id
                            
                            # Add all Inner Waypoints
                            for iwp in swp.get('inner_waypoints', []):
                                iwp_id = iwp.get('id', '')
                                iwp_name = iwp.get('name', iwp_id)
                                if iwp_name:
                                    self.available_waypoints.append(iwp_name)
                                    self.waypoint_id_to_name[iwp_id] = iwp_name
                                    self.waypoint_id_to_name[iwp_name] = iwp_name
                                    self.waypoint_name_to_id[iwp_name.lower()] = iwp_id
                    else:
                        logger.log_error('PlannerExecutor', 'Waypoint file does not have 2D format (super_waypoints)')
                        return False
                else:
                    logger.log_error('PlannerExecutor', 'No waypoint file found')
                    return False
            
            # Add special mappings for START
            self.waypoint_id_to_name['START'] = 'start'
            self.waypoint_id_to_name['SWP_001'] = self.waypoint_id_to_name.get('SWP_001', 'start')
            
            logger.log_debug('PlannerExecutor', f'Loaded {len(self.available_waypoints)} waypoints: {self.available_waypoints}')
            return len(self.available_waypoints) > 0
            
        except Exception as e:
            logger.log_error('PlannerExecutor', f'Failed to load waypoints: {e}')
            return False
    
    def _find_waypoint_file(self) -> Optional[str]:
        """Find the waypoint file path."""
        if self.waypoint_file:
            # Check in waypoint_dir or home directory
            if self.waypoint_dir:
                path = os.path.join(self.waypoint_dir, self.waypoint_file)
                if os.path.exists(path):
                    return path
            
            home_path = os.path.join(
                os.path.expanduser("~"), 
                "dronebuddylib", 
                "tellowaypoints",
                self.waypoint_file
            )
            if os.path.exists(home_path):
                return home_path
        
        return None
    
    def _execute_plan(self) -> PlannerSessionResult:
        """Execute the current action plan."""
        self.state = PlannerState.EXECUTING
        
        if not self.current_plan or not self.current_plan.actions:
            return self._create_error_result("No actions in plan")
        
        # Pre-warm YOLO-World model BEFORE takeoff if searching for non-COCO objects
        # This prevents drone timeout due to slow model loading during flight
        if not self.current_plan.is_coco_class:
            detection_targets = self.current_plan.get_detection_targets()
            self._send_message_to_user(
                'Preparing YOLO-World model for open-vocabulary detection... (this may take a moment)'
            )
            logger.log_info('PlannerExecutor', 
                f'Pre-warming YOLO-World model for non-COCO targets: {detection_targets}')
            
            prewarm_success = self.nav_engine.prewarm_yolo_world(
                target_objects=detection_targets,
                yolo_world_model_path=self.yolo_world_model_path
            )
            
            if prewarm_success:
                self._send_message_to_user('YOLO-World model ready!')
            else:
                logger.log_warning('PlannerExecutor', 
                    'Failed to pre-warm YOLO-World model. Detection may be slow during flight.')
                self._send_message_to_user(
                    'Warning: Could not pre-load detection model. Scan may be slow.'
                )
        
        # Takeoff
        self._send_message_to_user('Taking off...')
        try:
            self.nav_engine.takeoff()
        except Exception as e:
            logger.log_warning('PlannerExecutor', f'Takeoff note: {e}')
        
        # NavigationEngine now handles both navigation AND scanning with detection
        # via its unified scan_with_detection() method
        
        action_index = 0
        
        while action_index < len(self.current_plan.actions):
            action = self.current_plan.actions[action_index]
            
            logger.log_info('PlannerExecutor', 
                f'Executing action {action_index + 1}/{len(self.current_plan.actions)}: {action.action_type.value}')
            
            result = self._execute_action(action)
            
            if result == "OBJECT_FOUND":
                # Object found and confirmed - end session successfully
                self.state = PlannerState.OBJECT_FOUND
                self._safe_return_to_start()
                return self._create_success_result()
            
            elif result == "OBJECT_REJECTED":
                # User rejected the found object - need to replan
                return self._handle_rejection()
            
            elif result == "NO_CONFIRMATION_WANTED":
                # User doesn't want confirmation - end session
                self.state = PlannerState.COMPLETED
                self._safe_return_to_start()
                return self._create_partial_result("Search completed, no confirmation requested")
            
            elif result == "DRONE_LANDED_EARLY":
                # Drone landed during navigation - check if it was obstacle timeout
                self.state = PlannerState.ERROR
                
                # Check if it was specifically an obstacle timeout
                if TelloWaypointNavCoordinator._obstacle_timeout_occurred:
                    self._send_message_to_user(
                        "⚠️ Path blocked by obstacle for over 30 seconds. "
                        "Drone landed safely. Search session ended."
                    )
                    TelloWaypointNavCoordinator._obstacle_timeout_occurred = False  # Reset flag
                    return self._create_error_result(
                        "Search aborted: path blocked by obstacle for over 30 seconds"
                    )
                else:
                    self._send_message_to_user(
                        "⚠️ Drone landed unexpectedly during navigation. "
                        "Search session ended for safety."
                    )
                    return self._create_error_result(
                        "Search aborted: drone landed unexpectedly during navigation"
                    )
            
            elif result == "CONTINUE":
                # Continue to next action
                action_index += 1
            
            elif result == "ERROR":
                logger.log_error('PlannerExecutor', 'Action execution failed')
                action_index += 1  # Skip and continue
            
            else:
                action_index += 1
        
        # All actions exhausted, object not found
        return self._handle_not_found()
    
    def _execute_action(self, action: PlannerAction) -> str:
        """
        Execute a single action from the plan.
        
        Returns:
            "CONTINUE" - continue to next action
            "OBJECT_FOUND" - object found and confirmed
            "OBJECT_REJECTED" - object found but rejected by user
            "NO_CONFIRMATION_WANTED" - user doesn't want confirmation
            "ERROR" - action failed
        """
        try:
            if action.action_type == PlannerActionType.NAVIGATE_TO_WAYPOINT:
                return self._action_navigate(action.waypoint_name)
            
            elif action.action_type == PlannerActionType.SCAN_AREA:
                return self._action_scan()
            
            elif action.action_type == PlannerActionType.RETURN_TO_START:
                return self._action_return_to_start()
            
            elif action.action_type == PlannerActionType.LAND:
                return self._action_land()
            
            elif action.action_type == PlannerActionType.TAKEOFF:
                return "CONTINUE"  # Already took off
            
            else:
                logger.log_warning('PlannerExecutor', f'Unknown action type: {action.action_type}')
                return "CONTINUE"
                
        except Exception as e:
            logger.log_error('PlannerExecutor', f'Action execution error: {e}')
            return "ERROR"
    
    def _action_navigate(self, waypoint_name: str) -> str:
        """Execute navigation to a waypoint."""
        if not waypoint_name:
            logger.log_warning('PlannerExecutor', 'No waypoint specified for navigation')
            return "CONTINUE"
        
        # Display user-friendly name
        display_name = self._get_waypoint_display_name(waypoint_name)
        self._send_message_to_user(f'Flying to {display_name}...')
        
        try:
            result = self.nav_engine.navigate_to_waypoint(
                waypoint_name, 
                NavigationInstruction.CONTINUE
            )
            
            if result and len(result) >= 2:
                landed = result[0]  # True = drone has landed, False = still flying
                self.current_waypoint = result[1]
                
                # Check if drone landed unexpectedly (obstacle timeout, emergency, etc.)
                if landed:
                    logger.log_error('PlannerExecutor', 
                        f'Navigation failed: drone landed unexpectedly at {result[1]}')
                    return "DRONE_LANDED_EARLY"
                
                # Track visited with display name for user-friendly output
                if display_name not in self.waypoints_visited:
                    self.waypoints_visited.append(display_name)
                    
                logger.log_success('PlannerExecutor', f'Arrived at {waypoint_name}')
                return "CONTINUE"
            else:
                logger.log_warning('PlannerExecutor', f'Navigation result unclear')
                return "CONTINUE"
                
        except Exception as e:
            logger.log_error('PlannerExecutor', f'Navigation failed: {e}')
            return "ERROR"
    
    def _action_scan(self) -> str:
        """
        Execute scan operation at current waypoint using appropriate detection method.
        
        Uses dual detection system based on VLM classification:
        - COCO class objects: Uses scan_with_detection() with YOLO ONNX model
        - Non-COCO objects: Uses scan_with_any_detection() with YOLO-World PyTorch model
        """
        if not self.nav_engine:
            logger.log_error('PlannerExecutor', 'NavigationEngine not initialized')
            return "ERROR"
        
        # Use display name for user message
        display_name = self._get_waypoint_display_name(self.current_waypoint)
        self._send_message_to_user(f'Scanning area at {display_name}...')
        self.scans_performed += 1
        
        try:
            # Choose detection method based on VLM's classification
            if self.current_plan.is_coco_class:
                # Use standard YOLO ONNX for COCO 80-class objects
                logger.log_debug('PlannerExecutor', 
                    f'Using YOLO ONNX detection for COCO class: {self.target_object}')
                
                scan_result = self.nav_engine.scan_with_detection(
                    target_object=self.target_object,
                    yolo_model_path=self.yolo_model_path,
                    yolo_conf_threshold=self.yolo_conf_threshold,
                    yolo_iou_threshold=self.yolo_iou_threshold
                )
            else:
                # Use YOLO-World for open-vocabulary detection
                detection_targets = self.current_plan.get_detection_targets()
                logger.log_debug('PlannerExecutor', 
                    f'Using YOLO-World detection for targets: {detection_targets}')
                
                # Verify YOLO-World model path is set
                if not self.yolo_world_model_path:
                    logger.log_error('PlannerExecutor', 
                        'YOLO-World model path not configured for non-COCO object detection')
                    self._send_message_to_user(
                        'Error: YOLO-World model not configured for open-vocabulary detection.'
                    )
                    return "ERROR"
                
                scan_result = self.nav_engine.scan_with_any_detection(
                    target_objects=detection_targets,
                    yolo_world_model_path=self.yolo_world_model_path,
                    yolo_conf_threshold=self.yolo_world_conf_threshold
                )
            
            # Report what was found
            if scan_result.all_unique_objects:
                self._send_message_to_user(
                    f'I detected these objects: {", ".join(scan_result.all_unique_objects)}'
                )
            else:
                self._send_message_to_user('No objects detected at this location.')
            
            # Check if target was found
            if scan_result.target_object_found:
                return self._handle_object_found(scan_result)
            else:
                self._send_message_to_user(f'"{self.target_object}" not found here. Continuing search...')
                return "CONTINUE"
                
        except Exception as e:
            logger.log_error('PlannerExecutor', f'Scan failed: {e}')
            return "ERROR"
    
    def _handle_object_found(self, scan_result: ScanResult) -> str:
        """Handle when target object is detected during scan."""
        self.state = PlannerState.AWAITING_CONFIRMATION
        
        # Use display name for user-friendly message
        display_name = self._get_waypoint_display_name(self.current_waypoint)
        self._send_message_to_user(
            f' I found "{self.target_object}" at {display_name}!'
        )
        
        # Send prompt as message BEFORE calling callback (fixes ordering)
        self._send_message_to_user("Would you like me to describe this item for confirmation? (yes/no):")
        
        # Ask if user wants confirmation
        response = self.user_input_callback(
            "Would you like me to describe this item for confirmation? (yes/no): "
        ).strip().lower()
        
        if response in ['yes', 'y']:
            # Get VLM description of the object
            image_paths = scan_result.get_image_paths_with_target()
            
            # Log which images are being sent to VLM
            logger.log_info('PlannerExecutor', 
                f'Sending {len(image_paths)} image(s) to VLM for description:')
            for i, path in enumerate(image_paths):
                logger.log_info('PlannerExecutor', f'  Image {i+1}: {path}')
            
            if image_paths:
                self._send_message_to_user('Analyzing the images...')
                
                description = self.planner_agent.describe_object_in_images(
                    target_object=self.target_object,
                    image_paths=image_paths
                )
                
                if description:
                    self._send_message_to_user(
                        f'Here is what I found:\n{description.get("object_description", "")}'
                    )
                    
                    if description.get("visual_characteristics"):
                        chars = ", ".join(description["visual_characteristics"])
                        self._send_message_to_user(f'Key features: {chars}')
                    
                    if description.get("location_context"):
                        self._send_message_to_user(f'Location: {description["location_context"]}')
            
            # Send prompt as message BEFORE calling callback (fixes ordering)
            self._send_message_to_user("Is this the item you were looking for? (yes/no):")
            
            # Ask for final confirmation
            confirm = self.user_input_callback(
                "Is this the item you were looking for? (yes/no): "
            ).strip().lower()
            
            if confirm in ['yes', 'y']:
                self._send_message_to_user('Great! Returning to start position.')
                return "OBJECT_FOUND"
            else:
                self._send_message_to_user('I understand. This is not the correct item.')
                return "OBJECT_REJECTED"
        else:
            # User doesn't want VLM description - ask for direct confirmation
            display_name = self._get_waypoint_display_name(self.current_waypoint)
            self._send_message_to_user(
                f'I found a "{self.target_object}" at {display_name}.'
            )
            
            # Send prompt as message BEFORE calling callback (fixes ordering)
            self._send_message_to_user("Is this the item you were looking for? (yes/no):")
            
            confirm = self.user_input_callback(
                "Is this the item you were looking for? (yes/no): "
            ).strip().lower()
            
            if confirm in ['yes', 'y']:
                self._send_message_to_user('Great! Returning to start position.')
                return "OBJECT_FOUND"
            else:
                self._send_message_to_user('I understand. This is not the correct item.')
                return "OBJECT_REJECTED"
    
    def _handle_rejection(self) -> PlannerSessionResult:
        """Handle when user rejects the found object."""
        self.replan_count += 1
        
        if self.replan_count >= self.MAX_REPLAN_ATTEMPTS:
            self._send_message_to_user(
                'I have exhausted my re-planning attempts. Returning to start.'
            )
            self.state = PlannerState.OBJECT_NOT_FOUND
            self._safe_return_to_start()
            return self._create_failure_result("Maximum replan attempts reached")
        
        # Send prompt as message BEFORE calling callback (fixes ordering)
        self._send_message_to_user("Would you like me to continue searching? (yes/no):")
        
        # Ask if user wants to continue
        response = self.user_input_callback(
            "Would you like me to continue searching? (yes/no): "
        ).strip().lower()
        
        if response in ['yes', 'y']:
            # Send prompt as message BEFORE calling callback (fixes ordering)
            self._send_message_to_user("Any additional hints about what I should look for? (or press Enter to skip):")
            
            feedback = self.user_input_callback(
                "Any additional hints about what I should look for? (or press Enter to skip): "
            ).strip()
            
            self._send_message_to_user('Generating a new search plan...')
            
            # Regenerate plan with detection mode context
            new_plan = self.planner_agent.regenerate_plan(
                target_object=self.target_object,
                waypoint_names=self.available_waypoints,
                current_waypoint=self.current_waypoint,
                visited_waypoints=self.waypoints_visited,
                user_feedback=feedback,
                is_coco_class=self.current_plan.is_coco_class,
                related_objects=self.current_plan.related_objects
            )
            
            if new_plan:
                self.current_plan = new_plan
                self._send_message_to_user(f'New plan: {new_plan.reasoning}')
                return self._execute_plan()
            else:
                self._send_message_to_user('Failed to generate new plan. Returning to start.')
                self._safe_return_to_start()
                return self._create_error_result("Failed to regenerate plan")
        else:
            self.state = PlannerState.COMPLETED
            self._safe_return_to_start()
            return self._create_partial_result("User ended search session")
    
    def _handle_not_found(self) -> PlannerSessionResult:
        """Handle when all planned actions are exhausted without finding object."""
        self.replan_count += 1
        
        if self.replan_count >= self.MAX_REPLAN_ATTEMPTS:
            self._send_message_to_user(
                f'I have searched all locations but could not find "{self.target_object}". '
                'Returning to start position.'
            )
            self.state = PlannerState.OBJECT_NOT_FOUND
            self._safe_return_to_start()
            return self._create_failure_result("Object not found after exhaustive search")
        
        self._send_message_to_user(
            f'I have visited all planned locations but did not find "{self.target_object}".'
        )
        
        # Send prompt as message BEFORE calling callback (fixes ordering)
        self._send_message_to_user("Would you like me to search again? (yes/no):")
        
        # Ask if user wants to continue
        response = self.user_input_callback(
            "Would you like me to search again? (yes/no): "
        ).strip().lower()
        
        if response in ['yes', 'y']:
            self._send_message_to_user('Generating a new search plan...')
            
            # Regenerate plan with detection mode context
            new_plan = self.planner_agent.regenerate_plan(
                target_object=self.target_object,
                waypoint_names=self.available_waypoints,
                current_waypoint=self.current_waypoint,
                visited_waypoints=self.waypoints_visited,
                user_feedback="Previous search did not find the object. Please try different locations or re-scan.",
                is_coco_class=self.current_plan.is_coco_class,
                related_objects=self.current_plan.related_objects
            )
            
            if new_plan:
                self.current_plan = new_plan
                self._send_message_to_user(f'New plan: {new_plan.reasoning}')
                return self._execute_plan()
            else:
                self._send_message_to_user('Failed to generate new plan. Returning to start.')
                self._safe_return_to_start()
                return self._create_error_result("Failed to regenerate plan")
        else:
            self.state = PlannerState.COMPLETED
            self._safe_return_to_start()
            return self._create_failure_result("Search ended by user")
    
    def _action_return_to_start(self) -> str:
        """Execute return to start waypoint."""
        return self._safe_return_to_start()
    
    def _action_land(self) -> str:
        """Execute landing."""
        try:
            self.nav_engine.land()
            return "CONTINUE"
        except Exception as e:
            logger.log_error('PlannerExecutor', f'Landing failed: {e}')
            return "ERROR"
    
    def _safe_return_to_start(self) -> str:
        """Safely return drone to START waypoint."""
        self.state = PlannerState.RETURNING
        self._send_message_to_user('Returning to start position...')
        
        try:
            # Check if session was terminated (obstacle timeout, keyboard interrupt)
            # If so, don't try to navigate - drone is already landed
            if TelloWaypointNavCoordinator._session_terminated:
                logger.log_info('PlannerExecutor', 
                    'Session terminated - skipping return to start (drone already landed)')
                # Check if it was obstacle timeout
                if TelloWaypointNavCoordinator._obstacle_timeout_occurred:
                    self._send_message_to_user(
                        'Path was blocked by obstacle for over 30 seconds. Drone has landed safely.'
                    )
                    TelloWaypointNavCoordinator._obstacle_timeout_occurred = False
                else:
                    self._send_message_to_user('Drone has already landed safely.')
                return "CONTINUE"
            
            # Navigate to START (first Super Waypoint) with HALT instruction
            start_waypoint = "START"
            if self.available_waypoints and "START" not in self.available_waypoints:
                # Fallback to first available waypoint if START not found
                start_waypoint = self.available_waypoints[0]
            
            result = self.nav_engine.navigate_to_waypoint(
                start_waypoint,
                NavigationInstruction.HALT
            )
            
            # Check if navigation returned with landed=True (obstacle timeout or other issue)
            if result and len(result) >= 2:
                landed_early = result[0]
                actual_waypoint = result[1]
                
                # Only show "could not reach" if drone landed at a DIFFERENT waypoint
                # Note: START maps to SWP_001, so check both
                reached_start = (actual_waypoint == start_waypoint or 
                                actual_waypoint == "SWP_001" or 
                                actual_waypoint == "START")
                
                if landed_early and not reached_start:
                    # Use display name for user-friendly message
                    display_name = self._get_waypoint_display_name(actual_waypoint)
                    
                    # Drone landed before reaching START
                    if TelloWaypointNavCoordinator._obstacle_timeout_occurred:
                        self._send_message_to_user(
                            f'[STATUS]Path blocked by obstacle for over 30 seconds while returning. '
                            f'Drone landed safely near {display_name}.'
                        )
                        TelloWaypointNavCoordinator._obstacle_timeout_occurred = False
                    else:
                        self._send_message_to_user(
                            f'[STATUS]Could not reach start position - drone landed safely near {display_name}.'
                        )
                    self.current_waypoint = actual_waypoint
                    return "CONTINUE"
            
            self.current_waypoint = start_waypoint
            self._send_message_to_user('Landed safely at start position.')
            return "CONTINUE"
        except Exception as e:
            logger.log_error('PlannerExecutor', f'Return to start failed: {e}')
            # Try emergency land
            try:
                drone = self.nav_engine.get_drone_instance()
                if drone:
                    drone.land()
            except:
                pass
            return "ERROR"
    
    def _create_success_result(self) -> PlannerSessionResult:
        """Create a successful session result."""
        duration = time.time() - self.session_start_time
        return PlannerSessionResult(
            success=True,
            target_object=self.target_object,
            found_at_waypoint=self.current_waypoint,
            waypoints_visited=self.waypoints_visited,
            scans_performed=self.scans_performed,
            session_duration=duration,
            final_state=PlannerState.OBJECT_FOUND
        )
    
    def _create_failure_result(self, message: str) -> PlannerSessionResult:
        """Create a failed session result."""
        duration = time.time() - self.session_start_time
        return PlannerSessionResult(
            success=False,
            target_object=self.target_object,
            waypoints_visited=self.waypoints_visited,
            scans_performed=self.scans_performed,
            session_duration=duration,
            final_state=PlannerState.OBJECT_NOT_FOUND,
            error_message=message
        )
    
    def _create_partial_result(self, message: str) -> PlannerSessionResult:
        """Create a partial completion result."""
        duration = time.time() - self.session_start_time
        return PlannerSessionResult(
            success=False,
            target_object=self.target_object,
            waypoints_visited=self.waypoints_visited,
            scans_performed=self.scans_performed,
            session_duration=duration,
            final_state=PlannerState.COMPLETED,
            error_message=message
        )
    
    def _create_error_result(self, error_message: str) -> PlannerSessionResult:
        """Create an error session result."""
        duration = time.time() - self.session_start_time
        return PlannerSessionResult(
            success=False,
            target_object=self.target_object,
            waypoints_visited=self.waypoints_visited,
            scans_performed=self.scans_performed,
            session_duration=duration,
            final_state=PlannerState.ERROR,
            error_message=error_message
        )

    def _validate_yolo_class(self, target_object: str) -> bool:
        """
        Validate that the target object is a recognized YOLO COCO class.
        
        Args:
            target_object: The object class name to validate
            
        Returns:
            True if the object is a valid YOLO class, False otherwise
        """
        if not target_object:
            return False
        
        target_lower = target_object.lower().strip()
        
        # Special case: UNRECOGNIZED is handled separately
        if target_lower == "unrecognized":
            return False
        
        # Check exact match only - VLM should provide exact class names
        return target_lower in YOLO_COCO_CLASSES
    
    @staticmethod
    def get_supported_yolo_classes() -> list:
        """
        Get the list of all object classes that YOLO can detect.
        
        Use this to inform users what objects the system can search for.
        
        Returns:
            List of 80 COCO class names
        """
        return YOLO_COCO_CLASSES.copy()

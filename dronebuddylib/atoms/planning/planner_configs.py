"""
Planner Configuration Module

This module provides configuration classes for the VLM-based Planner system.
Users can modify API keys, model paths, and other settings here rather than
typing them each time when using the planner.

Supports multiple VLM providers:
- OpenAI (GPT-4, GPT-4o, GPT-5)
- Anthropic (Claude)
- Google (Gemini)

Usage:
    from dronebuddylib.atoms.planning import PlannerConfigs
    
    # OpenAI example
    config = PlannerConfigs(
        vlm_provider="openai",
        vlm_api_key="sk-...",
        vlm_model="gpt-4o",
        yolo_model_path="/path/to/yolo.onnx",
        waypoint_file_path="/path/to/waypoints.json"
    )
    
    # Anthropic example  
    config = PlannerConfigs(
        vlm_provider="anthropic",
        vlm_api_key="sk-ant-...",
        vlm_model="claude-3-5-sonnet-20241022",
        yolo_model_path="/path/to/yolo.onnx"
    )
    
    # Google example
    config = PlannerConfigs(
        vlm_provider="google",
        vlm_api_key="AIza...",
        vlm_model="gemini-1.5-pro",
        yolo_model_path="/path/to/yolo.onnx"
    )
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal
import os


VLM_PROVIDER_TYPE = Literal["openai", "anthropic", "google"]


@dataclass
class PlannerConfigs:
    """
    Configuration settings for the VLM-based Planner system.
    
    This class holds all configurable parameters for the planner including
    API credentials, model paths, and operational settings.
    
    Attributes:
        vlm_provider: The VLM provider to use ("openai", "anthropic", "google")
        vlm_api_key: API key for the VLM provider
        vlm_model: Model name (provider-specific, e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        vlm_temperature: Temperature setting for response randomness (0.0 - 1.0)
        yolo_model_path: Path to the YOLO ONNX model file for COCO 80-class object detection
        yolo_confidence_threshold: Minimum confidence for YOLO detections (0.0 - 1.0)
        yolo_iou_threshold: IOU threshold for NMS filtering (0.0 - 1.0)
        yolo_world_model_path: Path to YOLO-World .pt model for open-vocabulary detection (required)
        yolo_world_confidence_threshold: Confidence threshold for YOLO-World (default: 0.025)
        midas_model_path: Path to MiDaS ONNX model for depth-based obstacle detection
        obstacle_detection_mode: Obstacle sensitivity mode (OFF, LOW, MEDIUM, HIGH, VERY_HIGH)
        waypoint_file_path: Path to the waypoint definition JSON file (REQUIRED)
        waypoint_directory: Directory containing waypoint files
        available_waypoints: List of available waypoint names (auto-loaded from file if not provided)
        scan_image_directory: Directory to save scan images (default: ~/dronebuddylib/scans)
        max_replan_attempts: Maximum number of replanning attempts when object not found (default: 2)
        logger_location: Path for log files
        
    Example:
        >>> config = PlannerConfigs(
        ...     vlm_provider="openai",
        ...     vlm_api_key="sk-...",
        ...     vlm_model="gpt-4o",
        ...     yolo_model_path="models/yolov8n.onnx",
        ...     yolo_world_model_path="models/yolov8m-worldv2.pt",
        ...     midas_model_path="models/midas_v21_small_256.onnx",
        ...     obstacle_detection_mode="MEDIUM",
        ...     waypoint_file_path="waypoints/home.json"
        ... )
    """
    
    # VLM Provider Settings (provider-agnostic)
    vlm_provider: VLM_PROVIDER_TYPE = "openai"
    vlm_api_key: str = ""
    vlm_model: str = ""  # Empty = use provider default
    vlm_temperature: float = 0.3  # Lower temperature for more deterministic planning
    
    # YOLO Detection Settings (for COCO 80-class objects)
    yolo_model_path: str = ""
    yolo_confidence_threshold: float = 0.25
    yolo_iou_threshold: float = 0.45
    
    # YOLO-World Settings (for open-vocabulary detection of non-COCO objects)
    yolo_world_model_path: str = ""  # Path to yolov8-worldv2.pt (required for non-COCO detection)
    yolo_world_confidence_threshold: float = 0.025  # Lower threshold due to YOLO-World's lower confidence
    
    # MiDaS Obstacle Detection Settings
    midas_model_path: str = ""  # Path to MiDaS ONNX model file for depth-based obstacle detection
    obstacle_detection_mode: str = "OFF"  # OFF, LOW, MEDIUM, HIGH, VERY_HIGH
    
    # Waypoint Settings
    waypoint_file_path: str = ""
    waypoint_directory: str = ""
    available_waypoints: List[str] = field(default_factory=list)
    
    # Operational Settings
    scan_image_directory: Optional[str] = None
    max_replan_attempts: int = 2
    takeoff_altitude_cm: int = 0  # Target altitude (cm) after takeoff (0 = no adjustment, 20-500 valid range)
    mission_pad_enabled: bool = False  # Enable mission pad alignment after each waypoint arrival
    
    # Logging Settings
    logger_location: str = ""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Set default model based on provider if not specified
        if not self.vlm_model:
            default_models = {
                "openai": "gpt-4o",
                "anthropic": "claude-3-5-sonnet-20241022",
                "google": "gemini-1.5-pro"
            }
            self.vlm_model = default_models.get(self.vlm_provider, "gpt-4o")
        
        if not self.vlm_api_key:
            import warnings
            warnings.warn(
                f"VLM API key not provided for {self.vlm_provider}. Set vlm_api_key before using the planner.",
                UserWarning
            )
        
        if self.yolo_model_path and not os.path.exists(self.yolo_model_path):
            import warnings
            warnings.warn(
                f"YOLO model path does not exist: {self.yolo_model_path}",
                UserWarning
            )
        
        if self.waypoint_file_path and not os.path.exists(self.waypoint_file_path):
            import warnings
            warnings.warn(
                f"Waypoint file path does not exist: {self.waypoint_file_path}",
                UserWarning
            )
        
        # Set default scan directory if not provided
        if self.scan_image_directory is None:
            home_dir = os.path.expanduser("~")
            self.scan_image_directory = os.path.join(home_dir, "dronebuddylib", "scans")
    
    def get_provider_display_name(self) -> str:
        """Get a human-readable provider name."""
        names = {
            "openai": "OpenAI",
            "anthropic": "Anthropic (Claude)",
            "google": "Google (Gemini)"
        }
        return names.get(self.vlm_provider, self.vlm_provider)
    
    def load_waypoints_from_file(self) -> List[str]:
        """
        Load available waypoint names from the 2D waypoint file.
        
        Extracts ALL waypoint names including:
        - Super Waypoints (e.g., START, Kitchen, Living Room)
        - Inner Waypoints (e.g., Kitchen Table, Sofa Area)
        
        Returns:
            List of waypoint names extracted from the waypoint file.
            
        Raises:
            FileNotFoundError: If waypoint file doesn't exist
            json.JSONDecodeError: If waypoint file is invalid JSON
        """
        if not self.waypoint_file_path or not os.path.exists(self.waypoint_file_path):
            return []
        
        import json
        try:
            with open(self.waypoint_file_path, 'r') as f:
                waypoint_data = json.load(f)
            
            self.available_waypoints = []
            
            # 2D hierarchical format - extract ALL waypoint names
            if isinstance(waypoint_data, dict) and 'super_waypoints' in waypoint_data:
                for swp in waypoint_data['super_waypoints']:
                    # Add Super Waypoint name
                    swp_name = swp.get('name', swp.get('id', ''))
                    if swp_name:
                        self.available_waypoints.append(swp_name)
                    
                    # Add all Inner Waypoint names
                    for iwp in swp.get('inner_waypoints', []):
                        iwp_name = iwp.get('name', iwp.get('id', ''))
                        if iwp_name:
                            self.available_waypoints.append(iwp_name)
            else:
                import warnings
                warnings.warn("Waypoint file does not have 2D format (super_waypoints). Please use format version 2.0.", UserWarning)
            
            return self.available_waypoints
            
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load waypoints from file: {e}", UserWarning)
            return []
    
    def validate(self) -> tuple:
        """
        Validate all configuration settings.
        
        Returns:
            Tuple of (is_valid: bool, errors: List[str])
        """
        errors = []
        
        if self.vlm_provider not in ["openai", "anthropic", "google"]:
            errors.append(f"Invalid VLM provider: {self.vlm_provider}. Use: openai, anthropic, google")
        
        if not self.vlm_api_key:
            errors.append(f"VLM API key is required for {self.vlm_provider}")
        
        if not self.yolo_model_path:
            errors.append("YOLO model path is required")
        elif not os.path.exists(self.yolo_model_path):
            errors.append(f"YOLO model file not found: {self.yolo_model_path}")
        
        if not self.yolo_world_model_path:
            errors.append("YOLO-World model path is required for non-COCO object detection")
        elif not os.path.exists(self.yolo_world_model_path):
            errors.append(f"YOLO-World model file not found: {self.yolo_world_model_path}")
        
        if not 0 <= self.vlm_temperature <= 1:
            errors.append("vlm_temperature must be between 0 and 1")
        
        if not 0 <= self.yolo_confidence_threshold <= 1:
            errors.append("yolo_confidence_threshold must be between 0 and 1")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'vlm_provider': self.vlm_provider,
            'vlm_api_key': '***' if self.vlm_api_key else '',  # Mask API key
            'vlm_model': self.vlm_model,
            'vlm_temperature': self.vlm_temperature,
            'yolo_model_path': self.yolo_model_path,
            'yolo_confidence_threshold': self.yolo_confidence_threshold,
            'yolo_iou_threshold': self.yolo_iou_threshold,
            'yolo_world_model_path': self.yolo_world_model_path,
            'yolo_world_confidence_threshold': self.yolo_world_confidence_threshold,
            'waypoint_file_path': self.waypoint_file_path,
            'waypoint_directory': self.waypoint_directory,
            'available_waypoints': self.available_waypoints,
            'scan_image_directory': self.scan_image_directory,
            'max_replan_attempts': self.max_replan_attempts,
            'takeoff_altitude_cm': self.takeoff_altitude_cm,
            'mission_pad_enabled': self.mission_pad_enabled,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'PlannerConfigs':
        """Create configuration from dictionary."""
        # Handle legacy 'openai_*' keys for backward compatibility
        if 'openai_api_key' in config_dict and 'vlm_api_key' not in config_dict:
            config_dict['vlm_api_key'] = config_dict.pop('openai_api_key')
            config_dict['vlm_provider'] = 'openai'
        if 'openai_model' in config_dict and 'vlm_model' not in config_dict:
            config_dict['vlm_model'] = config_dict.pop('openai_model')
        if 'openai_temperature' in config_dict and 'vlm_temperature' not in config_dict:
            config_dict['vlm_temperature'] = config_dict.pop('openai_temperature')
        
        # Filter out masked API keys
        if config_dict.get('vlm_api_key') == '***':
            config_dict.pop('vlm_api_key')
        
        # Only pass known fields
        known_fields = {
            'vlm_provider', 'vlm_api_key', 'vlm_model', 'vlm_temperature',
            'yolo_model_path', 'yolo_confidence_threshold', 'yolo_iou_threshold',
            'yolo_world_model_path', 'yolo_world_confidence_threshold',
            'waypoint_file_path', 'waypoint_directory', 'available_waypoints',
            'scan_image_directory', 'max_replan_attempts', 'logger_location',
            'takeoff_altitude_cm', 'mission_pad_enabled'
        }
        filtered_dict = {k: v for k, v in config_dict.items() if k in known_fields}
        
        return cls(**filtered_dict)
    
    @classmethod
    def from_json_file(cls, file_path: str) -> 'PlannerConfigs':
        """
        Load configuration from a JSON file.
        
        Args:
            file_path: Path to the JSON configuration file
            
        Returns:
            PlannerConfigs instance with loaded settings
        """
        import json
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save_to_json_file(self, file_path: str, include_api_key: bool = False):
        """
        Save configuration to a JSON file.
        
        Args:
            file_path: Path where to save the configuration
            include_api_key: Whether to include the API key (default False for security)
        """
        import json
        config_dict = self.to_dict()
        if include_api_key:
            config_dict['vlm_api_key'] = self.vlm_api_key
        
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


# Default configuration template - users can copy and modify
DEFAULT_PLANNER_CONFIG = PlannerConfigs(
    vlm_provider="openai",
    vlm_api_key="",  # Set your API key here
    vlm_model="gpt-4o",
    yolo_model_path="",  # Set path to your YOLO ONNX model
    waypoint_file_path="",  # Set path to your waypoint file
)

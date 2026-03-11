"""
Example usage of the VLM-based Planner for Drone Buddy Library.

This script demonstrates how to use the PlannerEngine to find objects
using a Tello drone with VLM-based intelligent planning.

Supports multiple VLM providers:
- OpenAI (GPT-4, GPT-4o, GPT-5)
- Anthropic (Claude-3.5-Sonnet, Claude-3-Opus)
- Google (Gemini-1.5-Pro, Gemini-1.5-Flash)

Detection System:
- COCO 80-class objects: Uses standard YOLO ONNX model (fast)
- Non-COCO objects: Uses YOLO-World PyTorch model (open-vocabulary)

Prerequisites:
1. API key for your VLM provider (OpenAI, Anthropic, or Google)
2. ONNX-exported YOLOv8 model (e.g., yolov8n_640x640.onnx)
3. Optional: YOLO-World model for non-COCO object detection (e.g., yolov8m-worldv2.pt)
4. Pre-mapped waypoints (created using NavigationEngine.map_location())
5. Connected Tello drone

Usage:
    python planner_example.py           # Launch with GUI (default)
    python planner_example.py --gui     # Launch with GUI (explicit)
    python planner_example.py --cli     # Launch with terminal interface
"""

import sys
import os

# Add the library to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dronebuddylib import EngineConfigurations, AtomicEngineConfigurations
from dronebuddylib.atoms.planning import PlannerEngine, PlannerSessionResult
from dronebuddylib.utils.logger import Logger

logger = Logger()


# =============================================================================
# Configuration - Modify these values for your setup
# =============================================================================

def get_default_config() -> EngineConfigurations:
    """
    Get the default configuration for the planner.
    Modify these values according to your setup.
    """
    # Import ObstacleDetectionMode for MiDaS configuration
    from dronebuddylib.models.enums import ObstacleDetectionMode
    
    config = EngineConfigurations({})
    
    # VLM Configuration
    # Options: "openai", "anthropic", "google"
    config.add_configuration(AtomicEngineConfigurations.PLANNER_VLM_PROVIDER, "openai")
    config.add_configuration(AtomicEngineConfigurations.PLANNER_VLM_API_KEY, os.environ.get("OPENAI_API_KEY", "your-api-key-here"))
    config.add_configuration(AtomicEngineConfigurations.PLANNER_VLM_MODEL, "gpt-5.2")
    
    # YOLO Configuration - Standard YOLO for COCO 80-class objects
    config.add_configuration(
        AtomicEngineConfigurations.PLANNER_YOLO_ONNX_MODEL_PATH, 
        "C:/Users/zheng/FYP/yolo-onnx-models/yolo11m_320x320.onnx"
    )
    
    # YOLO-World Configuration - For non-COCO objects (glasses, keys, etc.)
    config.add_configuration(
        AtomicEngineConfigurations.PLANNER_YOLO_WORLD_MODEL_PATH,
        "C:/Users/zheng/FYP/yolov8m-worldv2.pt"
    )
    config.add_configuration(
        AtomicEngineConfigurations.PLANNER_YOLO_WORLD_CONF_THRESHOLD,
        0.05
    )
    
    # MiDaS Obstacle Detection Configuration
    # This enables depth-based obstacle avoidance during navigation
    # The drone will stop and wait when obstacles are detected before forward movements
    config.add_configuration(
        AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_MIDAS_MODEL_PATH,
        'C:\\Users\\zheng\\FYP\\models\\midas_small_384x288.onnx'  # Path to MiDaS ONNX model
    )
    config.add_configuration(
        AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_OBSTACLE_DETECTION_MODE,
        ObstacleDetectionMode.OFF  # Options: OFF, LOW, MEDIUM, HIGH, VERY_HIGH
    )
    
    config.add_configuration(AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_TAKEOFF_ALTITUDE_CM, -1)
    config.add_configuration(AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_MISSION_PAD_ENABLED, False)
    
    # Waypoint Configuration
    config.add_configuration(
        AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_FILE, 
        "drone_movements_20260131_163225.json"
    )
    
    return config


# =============================================================================
# GUI Mode - Clean User Interface (Recommended)
# =============================================================================

def run_gui_mode():
    """
    Launch the planner with a graphical user interface.
    
    This provides a clean, chat-like interface that separates user interaction
    from debug logs. Recommended for most users.
    """
    from dronebuddylib.atoms.planning.planner_gui import PlannerGUIApp
    
    logger.log_info("PlannerExample", "Starting Planner with GUI...")
    
    config = get_default_config()
    
    # Create and run the GUI application
    app = PlannerGUIApp(config=config)
    app.run()


# =============================================================================
# CLI Mode - Terminal Interface (Original)
# =============================================================================


def run_cli_mode():
    """
    Launch the planner with terminal interface.
    
    This is the original interface that uses console input/output.
    Debug logs will appear mixed with user messages.
    """
    logger.log_info("PlannerExample", "Starting Planner with CLI...")
    interactive_find_object()


def test_find_object():
    """Test the VLM-based object finding functionality."""
    
    config = get_default_config()
    
    # Initialize the engine
    engine = PlannerEngine(config)
    
    logger.log_info("PlannerExample", "Planner engine initialized successfully")
    
    # Find an object
    result = engine.find_object("Find my coffee cup")
    
    return result


def test_find_object_with_waypoint_file():
    """Test object finding with a specific waypoint file."""
    
    config = EngineConfigurations({})
    
    # VLM configuration
    config.add_configuration(AtomicEngineConfigurations.PLANNER_VLM_PROVIDER, "openai")
    config.add_configuration(AtomicEngineConfigurations.PLANNER_VLM_API_KEY, "YOUR_API_KEY_HERE")
    config.add_configuration(AtomicEngineConfigurations.PLANNER_VLM_MODEL, "gpt-4o")
    
    # YOLO configuration
    config.add_configuration(
        AtomicEngineConfigurations.PLANNER_YOLO_ONNX_MODEL_PATH, 
        "C:/Users/zheng/FYP/yolo-onnx-models/yolo11m_320x320.onnx"
    )
    
    # Use a specific waypoint file
    config.add_configuration(
        AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_FILE, 
        "drone_movements_20250717_143431.json"
    )
    
    engine = PlannerEngine(config)
    
    logger.log_info("PlannerExample", "Planner engine initialized with waypoint file")
    
    result = engine.find_object("Find my laptop")
    
    return result


def test_find_object_anthropic():
    """Test object finding using Anthropic Claude."""
    
    config = EngineConfigurations({})
    
    # Use Anthropic Claude
    config.add_configuration(AtomicEngineConfigurations.PLANNER_VLM_PROVIDER, "anthropic")
    config.add_configuration(AtomicEngineConfigurations.PLANNER_VLM_API_KEY, "YOUR_ANTHROPIC_API_KEY")
    config.add_configuration(AtomicEngineConfigurations.PLANNER_VLM_MODEL, "claude-3-5-sonnet-20241022")
    
    # YOLO configuration
    config.add_configuration(
        AtomicEngineConfigurations.PLANNER_YOLO_ONNX_MODEL_PATH, 
        "yolo-onnx-models/yolov8n_640x640.onnx"
    )
    
    # Required: Waypoint file
    config.add_configuration(
        AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_FILE, 
        "your_waypoints.json"
    )
    
    engine = PlannerEngine(config)
    
    logger.log_info("PlannerExample", "Planner engine initialized with Anthropic Claude")
    
    result = engine.find_object("Find my phone")
    
    return result


def test_find_object_google():
    """Test object finding using Google Gemini."""
    
    config = EngineConfigurations({})
    
    # Use Google Gemini
    config.add_configuration(AtomicEngineConfigurations.PLANNER_VLM_PROVIDER, "google")
    config.add_configuration(AtomicEngineConfigurations.PLANNER_VLM_API_KEY, "YOUR_GOOGLE_API_KEY")
    config.add_configuration(AtomicEngineConfigurations.PLANNER_VLM_MODEL, "gemini-1.5-pro")
    
    # YOLO configuration
    config.add_configuration(
        AtomicEngineConfigurations.PLANNER_YOLO_ONNX_MODEL_PATH, 
        "yolo-onnx-models/yolov8n_640x640.onnx"
    )
    
    # Required: Waypoint file
    config.add_configuration(
        AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_FILE, 
        "your_waypoints.json" 
    )
    
    engine = PlannerEngine(config)
    
    logger.log_info("PlannerExample", "Planner engine initialized with Google Gemini")
    
    result = engine.find_object("Find my keys")
    
    return result


def interactive_find_object():
    """Interactive mode for finding objects (terminal-based)."""
    
    config = get_default_config()
    
    engine = PlannerEngine(config)
    
    logger.log_info("PlannerExample", "Planner engine initialized for interactive mode")
    
    print("=" * 60)
    print("  VLM-based Drone Object Finder (Terminal Mode)")
    print("  Type 'quit' or 'exit' to stop")
    print("=" * 60)
    
    while True:
        print()
        user_request = input(" What would you like me to find? ").strip()
        
        if user_request.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! ")
            break
        
        if not user_request:
            print("Please enter an object to find.")
            continue
        
        print(f"\n Starting search for: '{user_request}'")
        print("-" * 50)
        
        try:
            result: PlannerSessionResult = engine.find_object(user_request)
            
            print("\n" + "=" * 50)
            print("  SESSION RESULTS")
            print("=" * 50)
            
            if result.success:
                print(f" SUCCESS! Found '{result.target_object}'")
                print(f"   Location: {result.found_at_waypoint}")
            else:
                print(f" Could not find '{result.target_object}'")
                if result.error_message:
                    print(f"   Reason: {result.error_message}")
            
            print(f"\n Session Statistics:")
            print(f"   - Waypoints visited: {', '.join(result.waypoints_visited) or 'None'}")
            print(f"   - Scans performed: {result.scans_performed}")
            print(f"   - Duration: {result.session_duration:.1f} seconds")
            print(f"   - Final state: {result.final_state.value}")
            
        except KeyboardInterrupt:
            print("\n\n Session interrupted by user")
            break
        except Exception as e:
            print(f"\n Error during execution: {e}")
            logger.log_error('PlannerExample', f'Execution error: {e}')


def main():
    """Main function - parse arguments and run appropriate mode."""
    
    # Parse command line arguments
    use_gui = True  # Default to GUI mode
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--cli', '-c', '--terminal', '-t']:
            use_gui = False
        elif arg in ['--gui', '-g']:
            use_gui = True
        elif arg in ['--help', '-h']:
            print(__doc__)
            print("\nOptions:")
            print("  --gui, -g      Launch with graphical interface (default)")
            print("  --cli, -c      Launch with terminal interface")
            print("  --help, -h     Show this help message")
            sys.exit(0)
    
    logger.log_info("PlannerExample", f"Starting VLM Planner ({'GUI' if use_gui else 'CLI'} mode)")
    
    if use_gui:
        run_gui_mode()
    else:
        run_cli_mode()


if __name__ == "__main__":
    main()

"""This is the main python module where we call all the function to do
something...
"""
import sys
import os
import time
import platform
from djitellopy import Tello

# Set platform-specific environment variables
if platform.system() == 'Linux':
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

import cv2

# Use cross-platform path
if platform.system() == 'Windows':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
else:
    sys.path.insert(0, '/home/zen/drone-buddy-library') 


from dronebuddylib import EngineConfigurations, NavigationAlgorithm, NavigationEngine, AtomicEngineConfigurations
from dronebuddylib.atoms.navigation import NavigationInstruction
from dronebuddylib.utils.logger import Logger

logger = Logger()

def test_mapping(): 
    """Test the mapping functionality of the NavigationEngine."""
    # Configure navigation engine
    config = EngineConfigurations({})
    engine = NavigationEngine(NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT, config)
    
    logger.log_info("Main", "Navigation engine initialized successfully")

    # Perform mapping
    result = engine.map_location()

    return result

def test_navigate(): 
    """Test the navigation functionality of the NavigationEngine."""
    # Configure navigation engine
    config = EngineConfigurations({})
    engine = NavigationEngine(NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT, config)
    
    logger.log_info("Main", "Navigation engine initialized successfully")

    # Perform navigation
    result = engine.navigate()

    return result

def test_navigate_with_waypoint_file(): 
    """Test the navigation functionality with a waypoint file."""
    # Configure navigation engine
    config = EngineConfigurations({})
    config.add_configuration(AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_FILE, 'drone_movements_20250717_143431.json')
    engine = NavigationEngine(NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT, config)
    
    logger.log_info("Main", "Navigation engine initialized successfully")

    # Perform navigation with waypoint file
    result = engine.navigate()

    return result

def test_navigate_to_waypoint_with_waypoint_file(): 
    """Test the navigate_to_waypoint function with a waypoint file."""
    # Configure navigation engine
    config = EngineConfigurations({})
    config.add_configuration(AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_FILE, 'drone_movements_20250717_143431.json')
    engine = NavigationEngine(NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT, config)

    logger.log_info("Main", "Navigation engine initialized successfully")

    # Using proper NavigationInstruction enum values
    result1 = engine.navigate_to_waypoint("WP_002", NavigationInstruction.CONTINUE)
    result2 = engine.navigate_to_waypoint("WP_001", NavigationInstruction.HALT)

    return result1, result2

def test_navigate_to_waypoint(): 
    """Test the navigate_to_waypoint function with proper NavigationInstruction enum usage."""
    # Configure navigation engine
    config = EngineConfigurations({})
    engine = NavigationEngine(NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT, config)
    
    logger.log_info("Main", "Navigation engine initialized successfully")

    # Using proper NavigationInstruction enum values
    result1 = engine.navigate_to_waypoint("WP_002", NavigationInstruction.CONTINUE)
    result2 = engine.navigate_to_waypoint("WP_001", NavigationInstruction.HALT)

    return result1, result2

def test_navigate_to():
    """Test the navigate_to function with proper NavigationInstruction enum usage."""
    # Configure navigation engine
    config = EngineConfigurations({})
    engine = NavigationEngine(NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT, config)
    
    logger.log_info("Main", "Navigation engine initialized successfully")

    result = engine.navigate_to(["WP_002", "WP_003", "Kitchen", "WP_001", "WP_002"], NavigationInstruction.HALT)

    return result

def test_navigate_to_with_waypoint_file():
    """Test the navigate_to function with a waypoint file."""
    # Configure navigation engine
    config = EngineConfigurations({})
    config.add_configuration(AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_FILE, 'drone_movements_20250717_143431.json')
    engine = NavigationEngine(NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT, config)
    
    logger.log_info("Main", "Navigation engine initialized successfully")

    result = engine.navigate_to(["WP_002", "WP_003", "Kitchen", "WP_001", "WP_002"], NavigationInstruction.HALT)

    return result

def test_navigate_to_waypoint_with_scan(): 
    """Test the navigate_to_waypoint function with scan functionality."""
    # Configure navigation engine
    config = EngineConfigurations({})
    engine = NavigationEngine(NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT, config)
    
    logger.log_info("Main", "Navigation engine initialized successfully")

    # Using proper NavigationInstruction enum values
    engine.navigate_to_waypoint("WP_002", NavigationInstruction.CONTINUE)
    images1 = engine.scan_surrounding()
    engine.navigate_to_waypoint("WP_001", NavigationInstruction.HALT)
    images2 = engine.scan_surrounding()
    return images1, images2

def main():
    """This is the main function we call when running the python file."""
    
    logger.log_info("Main", "Starting Tello Navigation Tests")
    print(test_navigate_to_with_waypoint_file())

if __name__ == "__main__":
    main()
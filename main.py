"""This is the main python module where we call all the function to do
something...
"""
import sys
import os
import time

sys.path.insert(0, '/home/zen/drone-buddy-library') 


from dronebuddylib import EngineConfigurations, NavigationAlgorithm, NavigationEngine
from dronebuddylib.atoms.navigation import NavigationInstruction
from dronebuddylib.utils.logger import Logger

logger = Logger()

def test_navigate_to_waypoint():
    """Test the navigate_to_waypoint function with proper NavigationInstruction enum usage."""
    # Configure navigation engine
    config = EngineConfigurations({})
    engine = NavigationEngine(NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT, config)
    
    logger.log_info("Main", "Navigation engine initialized successfully")

    # Using proper NavigationInstruction enum values
    result1 = engine.navigate_to_waypoint("WP_002", NavigationInstruction.CONTINUE)
    time.sleep(5)  # Wait for the first navigation to complete
    result2 = engine.navigate_to_waypoint("WP_003", NavigationInstruction.CONTINUE)
    time.sleep(5)  # Wait for the second navigation to complete
    result3 = engine.navigate_to_waypoint("WP_001", 1)

    return result1, result2, result3

    # return engine.navigate()

    # return engine.map_location()

def main():
    """This is the main function we call when running the python file."""
    
    logger.log_info("Main", "Starting Tello Navigation Tests")
    print(test_navigate_to_waypoint())

if __name__ == "__main__":
    main()
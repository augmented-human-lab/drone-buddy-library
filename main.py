"""This is the main python module where we call all the function to do
something...
"""
import sys
import os
import time

sys.path.insert(0, '/home/zen/drone-buddy-library') 


from dronebuddylib import EngineConfigurations, NavigationAlgorithm, NavigationEngine
from dronebuddylib.utils.logger import Logger

logger = Logger()

def test_navigate_to_waypoint():
    """Test the navigate_to_waypoint function with different scenarios."""
    # Configure navigation engine
    config = EngineConfigurations({})
    engine = NavigationEngine(NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT, config)
    
    logger.log_info("Main", "Navigation engine initialized successfully")

    result1 = engine.navigate_to_waypoint("WP_002", "continue")
    time.sleep(2)  # Wait for the first navigation to complete
    result2 = engine.navigate_to_waypoint("WP_001", "halt")

    return result1, result2

    # return engine.navigate()

    # return engine.map_location()

def main():
    """This is the main function we call when running the python file."""
    
    logger.log_info("Main", "Starting Tello Navigation Tests")
    print(test_navigate_to_waypoint())

if __name__ == "__main__":
    main()
    
    """This is the main python module where we call all the function to do
something...
"""


# def main():
#     """This is the main function we call when running the python file."""
#     pass


# if __name__ == "__main__":
#     main()
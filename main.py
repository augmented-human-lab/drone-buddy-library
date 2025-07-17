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

"""
"(venv) zen@soccf-isa3-001:~/drone-buddy-library$ python3 main.py

2025-07-17 14:44:06 : INFO :Main : Starting Tello Navigation Tests


2025-07-17 14:44:06 : INFO :NAVIGATION_ENGINE : Preparing to initialize Tello Waypoint navigation engine.


2025-07-17 14:44:06 : INFO :NAVIGATION_TELLO_WAYPOINT : Initializing Tello navigation engine.


2025-07-17 14:44:06 : DEBUG :NAVIGATION_TELLO_WAYPOINT : Waypoint directory set to: /home/zen/dronebuddylib/tellowaypoints


2025-07-17 14:44:06 : INFO :NAVIGATION_TELLO_WAYPOINT : Tello navigation engine initialized successfully.


2025-07-17 14:44:06 : DEBUG :NAVIGATION_TELLO_WAYPOINT : Configuration: vertical_factor=1.0, mapping_movement_speed=38, mapping_rotation_speed=70, nav_speed=55


2025-07-17 14:44:06 : DEBUG :NAVIGATION_ENGINE : Tello Waypoint navigation engine initialized successfully.


2025-07-17 14:44:06 : INFO :Main : Navigation engine initialized successfully


2025-07-17 14:44:06 : INFO :NAVIGATION_ENGINE : Starting navigation to waypoint: WP_002


2025-07-17 14:44:06 : DEBUG :NAVIGATION_ENGINE : Navigation instruction: continue


2025-07-17 14:44:06 : INFO :NAVIGATION_TELLO_WAYPOINT : Starting navigation to waypoint: WP_002


2025-07-17 14:44:06 : DEBUG :NAVIGATION_TELLO_WAYPOINT : Navigation instruction: continue


2025-07-17 14:44:06 : INFO :TelloWaypointNavCoordinator : Initializing coordinator in goto mode.


2025-07-17 14:44:06 : DEBUG :TelloWaypointNavCoordinator : Initializing Tello drone.

[INFO] tello.py - 129 - Tello instance was initialized. Host: '192.168.10.1'. Port: '8889'.
INFO:djitellopy:Tello instance was initialized. Host: '192.168.10.1'. Port: '8889'.

2025-07-17 14:44:06 : DEBUG :TelloWaypointNavCoordinator : Coordinator initialized with params: waypoint_dir=/home/zen/dronebuddylib/tellowaypoints, mode=goto, vertical_factor=1.0


2025-07-17 14:44:06 : INFO :TelloWaypointNavCoordinator : Starting navigation run in goto mode.


2025-07-17 14:44:06 : INFO :TelloWaypointNavCoordinator : GOTO MODE ACTIVATED


2025-07-17 14:44:06 : INFO :TelloWaypointNavCoordinator : Connecting to Tello drone...

[INFO] tello.py - 438 - Send command: 'command'
INFO:djitellopy:Send command: 'command'
[INFO] tello.py - 462 - Response command: 'ok'
INFO:djitellopy:Response command: 'ok'

2025-07-17 14:44:06 : SUCCESS :TelloWaypointNavCoordinator : Drone connected successfully!

[INFO] tello.py - 438 - Send command: 'battery?'
INFO:djitellopy:Send command: 'battery?'
[INFO] tello.py - 462 - Response battery?: '73'
INFO:djitellopy:Response battery?: '73'

2025-07-17 14:44:06 : INFO :TelloWaypointNavCoordinator : Battery: 73%


2025-07-17 14:44:06 : INFO :TelloWaypointNavCoordinator : Taking off...

[INFO] tello.py - 438 - Send command: 'takeoff'
INFO:djitellopy:Send command: 'takeoff'
[INFO] tello.py - 462 - Response takeoff: 'ok'
INFO:djitellopy:Response takeoff: 'ok'

2025-07-17 14:44:16 : SUCCESS :TelloWaypointNavCoordinator : Drone is airborne!

[INFO] tello.py - 438 - Send command: 'battery?'
INFO:djitellopy:Send command: 'battery?'

2025-07-17 14:44:16 : INFO :TelloWaypointNavCoordinator : Battery monitoring thread started.


2025-07-17 14:44:16 : INFO :WaypointNavigationManager : Initializing waypoint navigation manager.


2025-07-17 14:44:16 : DEBUG :WaypointNavigationManager : Initialized with nav_speed=55, vertical_factor=1.0


2025-07-17 14:44:16 : INFO :WaypointNavigationManager : Loading waypoint file: /home/zen/dronebuddylib/tellowaypoints/drone_movements_20250717_143431.json


2025-07-17 14:44:16 : SUCCESS :WaypointNavigationManager : Loaded 2 waypoints successfully.


📍 WAYPOINT SUMMARY
==================================================
🏠 CURRENT WP_001: 'START'
   WP_002: 'END'
==================================================

2025-07-17 14:44:16 : INFO :TelloWaypointNavCoordinator : Navigating to waypoint: WP_002


2025-07-17 14:44:16 : INFO :WaypointNavigationManager : Navigation plan: From WP_001 ("START") to WP_002 ("END") using forward direction with 3 movements


🧭 NAVIGATION PLAN
From: WP_001 ('START')
To: WP_002 ('END')
Direction: forward
Total movements: 3

2025-07-17 14:44:16 : INFO :WaypointNavigationManager : Executing 3 movements (forward)

[INFO] tello.py - 438 - Send command: 'speed 55'
INFO:djitellopy:Send command: 'speed 55'
[INFO] tello.py - 462 - Response battery?: '72'
INFO:djitellopy:Response battery?: '72'
[INFO] tello.py - 462 - Response speed 55: 'ok'
INFO:djitellopy:Response speed 55: 'ok'

2025-07-17 14:44:16 : DEBUG :WaypointNavigationManager : Step 1/3: move movement

[INFO] tello.py - 438 - Send command: 'attitude?'
INFO:djitellopy:Send command: 'attitude?'
[INFO] tello.py - 462 - Response attitude?: 'pitch:0;roll:0;yaw:0;'
INFO:djitellopy:Response attitude?: 'pitch:0;roll:0;yaw:0;'

2025-07-17 14:44:17 : DEBUG :WaypointNavigationManager : Raw attitude response: pitch:0;roll:0;yaw:0;


2025-07-17 14:44:17 : DEBUG :WaypointNavigationManager : Adjusting yaw from 0 to 85 degrees

[INFO] tello.py - 438 - Send command: 'cw 85'
INFO:djitellopy:Send command: 'cw 85'
[INFO] tello.py - 462 - Response cw 85: 'ok'
INFO:djitellopy:Response cw 85: 'ok'
[INFO] tello.py - 471 - Send command (no response expected): 'rc 0 0 0 0'
INFO:djitellopy:Send command (no response expected): 'rc 0 0 0 0'
[INFO] tello.py - 438 - Send command: 'forward 136'
INFO:djitellopy:Send command: 'forward 136'
[INFO] tello.py - 438 - Send command: 'battery?'
INFO:djitellopy:Send command: 'battery?'
[INFO] tello.py - 462 - Response battery?: '72'
INFO:djitellopy:Response battery?: '72'
[INFO] tello.py - 462 - Response forward 136: 'ok'
INFO:djitellopy:Response forward 136: 'ok'
[INFO] tello.py - 471 - Send command (no response expected): 'rc 0 0 0 0'
INFO:djitellopy:Send command (no response expected): 'rc 0 0 0 0'

2025-07-17 14:44:22 : DEBUG :WaypointNavigationManager : Moved forward 136.0 cm at yaw 85 degrees


2025-07-17 14:44:22 : DEBUG :WaypointNavigationManager : Step 2/3: move movement

[INFO] tello.py - 438 - Send command: 'attitude?'
INFO:djitellopy:Send command: 'attitude?'
[INFO] tello.py - 462 - Response attitude?: 'pitch:1;roll:0;yaw:84;'
INFO:djitellopy:Response attitude?: 'pitch:1;roll:0;yaw:84;'

2025-07-17 14:44:23 : DEBUG :WaypointNavigationManager : Raw attitude response: pitch:1;roll:0;yaw:84;


2025-07-17 14:44:23 : DEBUG :WaypointNavigationManager : Adjusting yaw from 84 to 160 degrees

[INFO] tello.py - 438 - Send command: 'cw 76'
INFO:djitellopy:Send command: 'cw 76'
[INFO] tello.py - 462 - Response cw 76: 'ok'
INFO:djitellopy:Response cw 76: 'ok'
[INFO] tello.py - 471 - Send command (no response expected): 'rc 0 0 0 0'
INFO:djitellopy:Send command (no response expected): 'rc 0 0 0 0'
[INFO] tello.py - 438 - Send command: 'forward 53'
INFO:djitellopy:Send command: 'forward 53'
[INFO] tello.py - 462 - Response forward 53: 'ok'
INFO:djitellopy:Response forward 53: 'ok'
[INFO] tello.py - 438 - Send command: 'battery?'
INFO:djitellopy:Send command: 'battery?'
[INFO] tello.py - 462 - Response battery?: '72'
INFO:djitellopy:Response battery?: '72'
[INFO] tello.py - 471 - Send command (no response expected): 'rc 0 0 0 0'
INFO:djitellopy:Send command (no response expected): 'rc 0 0 0 0'

2025-07-17 14:44:27 : DEBUG :WaypointNavigationManager : Moved forward 53.5 cm at yaw 160 degrees


2025-07-17 14:44:27 : DEBUG :WaypointNavigationManager : Step 3/3: move movement

[INFO] tello.py - 438 - Send command: 'attitude?'
INFO:djitellopy:Send command: 'attitude?'
[INFO] tello.py - 462 - Response attitude?: 'pitch:1;roll:1;yaw:158;'
INFO:djitellopy:Response attitude?: 'pitch:1;roll:1;yaw:158;'

2025-07-17 14:44:27 : DEBUG :WaypointNavigationManager : Raw attitude response: pitch:1;roll:1;yaw:158;


2025-07-17 14:44:27 : DEBUG :WaypointNavigationManager : Adjusting yaw from 158 to -83 degrees

[INFO] tello.py - 438 - Send command: 'cw 119'
INFO:djitellopy:Send command: 'cw 119'
[INFO] tello.py - 462 - Response cw 119: 'ok'
INFO:djitellopy:Response cw 119: 'ok'
[INFO] tello.py - 471 - Send command (no response expected): 'rc 0 0 0 0'
INFO:djitellopy:Send command (no response expected): 'rc 0 0 0 0'
[INFO] tello.py - 438 - Send command: 'forward 139'
INFO:djitellopy:Send command: 'forward 139'
[INFO] tello.py - 438 - Send command: 'battery?'
INFO:djitellopy:Send command: 'battery?'
[INFO] tello.py - 462 - Response forward 139: '70'
INFO:djitellopy:Response forward 139: '70'
[INFO] tello.py - 438 - Send command: 'forward 139'
INFO:djitellopy:Send command: 'forward 139'
[INFO] tello.py - 462 - Response battery?: 'ok'
INFO:djitellopy:Response battery?: 'ok'

2025-07-17 14:44:37 : WARNING :TelloWaypointNavCoordinator : Battery check failed: invalid literal for int() with base 10: 'ok'

[INFO] tello.py - 438 - Send command: 'battery?'
INFO:djitellopy:Send command: 'battery?'
[INFO] tello.py - 462 - Response battery?: '68'
INFO:djitellopy:Response battery?: '68'
[WARNING] tello.py - 448 - Aborting command 'forward 139'. Did not receive a response after 7 seconds
WARNING:djitellopy:Aborting command 'forward 139'. Did not receive a response after 7 seconds
[INFO] tello.py - 438 - Send command: 'forward 139'
INFO:djitellopy:Send command: 'forward 139'
[INFO] tello.py - 462 - Response forward 139: 'ok'
INFO:djitellopy:Response forward 139: 'ok'
[INFO] tello.py - 471 - Send command (no response expected): 'rc 0 0 0 0'
INFO:djitellopy:Send command (no response expected): 'rc 0 0 0 0'

2025-07-17 14:44:46 : DEBUG :WaypointNavigationManager : Moved forward 139.6 cm at yaw -83 degrees


2025-07-17 14:44:46 : SUCCESS :WaypointNavigationManager : Navigation movements completed


2025-07-17 14:44:46 : SUCCESS :WaypointNavigationManager : Successfully navigated to WP_002 ("END")


2025-07-17 14:44:46 : SUCCESS :TelloWaypointNavCoordinator : Reached waypoint "WP_002"


2025-07-17 14:44:46 : INFO :TelloWaypointNavCoordinator : Continuing at waypoint "WP_002"


2025-07-17 14:44:46 : INFO :NAVIGATION_TELLO_WAYPOINT : Navigation to waypoint session closed with drone at current waypoint: WP_002.


2025-07-17 14:44:46 : DEBUG :NAVIGATION_ENGINE : Navigate to waypoint operation completed with drone at current waypoint: WP_002.

[INFO] tello.py - 438 - Send command: 'battery?'
INFO:djitellopy:Send command: 'battery?'
[INFO] tello.py - 462 - Response battery?: '68'
INFO:djitellopy:Response battery?: '68'

2025-07-17 14:44:48 : INFO :NAVIGATION_ENGINE : Starting navigation to waypoint: WP_001


2025-07-17 14:44:48 : DEBUG :NAVIGATION_ENGINE : Navigation instruction: halt


2025-07-17 14:44:48 : INFO :NAVIGATION_TELLO_WAYPOINT : Starting navigation to waypoint: WP_001


2025-07-17 14:44:48 : DEBUG :NAVIGATION_TELLO_WAYPOINT : Navigation instruction: halt


2025-07-17 14:44:48 : INFO :TelloWaypointNavCoordinator : Starting navigation run in goto mode.


2025-07-17 14:44:48 : INFO :TelloWaypointNavCoordinator : GOTO MODE ACTIVATED


2025-07-17 14:44:48 : INFO :TelloWaypointNavCoordinator : Navigating to waypoint: WP_001


2025-07-17 14:44:48 : INFO :WaypointNavigationManager : Navigation plan: From WP_002 ("END") to WP_001 ("START") using reverse direction with 3 movements


🧭 NAVIGATION PLAN
From: WP_002 ('END')
To: WP_001 ('START')
Direction: reverse
Total movements: 3

2025-07-17 14:44:48 : INFO :WaypointNavigationManager : Executing 3 movements (reverse)

[INFO] tello.py - 438 - Send command: 'speed 55'
INFO:djitellopy:Send command: 'speed 55'
[INFO] tello.py - 462 - Response speed 55: 'ok'
INFO:djitellopy:Response speed 55: 'ok'

2025-07-17 14:44:48 : DEBUG :WaypointNavigationManager : Step 1/3: move movement

[INFO] tello.py - 438 - Send command: 'attitude?'
INFO:djitellopy:Send command: 'attitude?'
[INFO] tello.py - 462 - Response attitude?: 'pitch:0;roll:0;yaw:-84;'
INFO:djitellopy:Response attitude?: 'pitch:0;roll:0;yaw:-84;'

2025-07-17 14:44:48 : DEBUG :WaypointNavigationManager : Raw attitude response: pitch:0;roll:0;yaw:-84;


2025-07-17 14:44:48 : DEBUG :WaypointNavigationManager : Adjusting yaw from -84 to 97 degrees

[INFO] tello.py - 438 - Send command: 'ccw 179'
INFO:djitellopy:Send command: 'ccw 179'
[INFO] tello.py - 462 - Response ccw 179: 'ok'
INFO:djitellopy:Response ccw 179: 'ok'
[INFO] tello.py - 471 - Send command (no response expected): 'rc 0 0 0 0'
INFO:djitellopy:Send command (no response expected): 'rc 0 0 0 0'
[INFO] tello.py - 438 - Send command: 'forward 139'
INFO:djitellopy:Send command: 'forward 139'
[INFO] tello.py - 438 - Send command: 'battery?'
INFO:djitellopy:Send command: 'battery?'
[INFO] tello.py - 462 - Response forward 139: '66'
INFO:djitellopy:Response forward 139: '66'
[INFO] tello.py - 438 - Send command: 'forward 139'
INFO:djitellopy:Send command: 'forward 139'
[INFO] tello.py - 462 - Response battery?: 'ok'
INFO:djitellopy:Response battery?: 'ok'

2025-07-17 14:44:55 : WARNING :TelloWaypointNavCoordinator : Battery check failed: invalid literal for int() with base 10: 'ok'

[WARNING] tello.py - 448 - Aborting command 'forward 139'. Did not receive a response after 7 seconds
WARNING:djitellopy:Aborting command 'forward 139'. Did not receive a response after 7 seconds
[INFO] tello.py - 438 - Send command: 'forward 139'
INFO:djitellopy:Send command: 'forward 139'
[INFO] tello.py - 438 - Send command: 'battery?'
INFO:djitellopy:Send command: 'battery?'
[INFO] tello.py - 462 - Response forward 139: '65'
INFO:djitellopy:Response forward 139: '65'

2025-07-17 14:45:00 : ERROR : WaypointNavigationManager : Error during navigation execution: Command 'forward 139' was unsuccessful for 4 tries. Latest response:     '65'

[INFO] tello.py - 471 - Send command (no response expected): 'rc 0 0 0 0'
INFO:djitellopy:Send command (no response expected): 'rc 0 0 0 0'

2025-07-17 14:45:00 : ERROR : WaypointNavigationManager : Navigation to WP_001 failed


2025-07-17 14:45:00 : ERROR : TelloWaypointNavCoordinator : Failed to reach waypoint "WP_001"


2025-07-17 14:45:02 : INFO :TelloWaypointNavCoordinator : Battery monitoring thread stopped.


2025-07-17 14:45:02 : INFO :TelloWaypointNavCoordinator : Cleaning up resources...


2025-07-17 14:45:02 : INFO :TelloWaypointNavCoordinator : Landing drone during cleanup...

[INFO] tello.py - 438 - Send command: 'land'
INFO:djitellopy:Send command: 'land'
[INFO] tello.py - 462 - Response battery?: 'error Auto land'
INFO:djitellopy:Response battery?: 'error Auto land'

2025-07-17 14:45:03 : WARNING :TelloWaypointNavCoordinator : Battery check failed: invalid literal for int() with base 10: 'error Auto land'

[INFO] tello.py - 462 - Response land: 'ok'
INFO:djitellopy:Response land: 'ok'

2025-07-17 14:45:09 : INFO :TelloWaypointNavCoordinator : Disconnecting from drone...


2025-07-17 14:45:09 : SUCCESS :TelloWaypointNavCoordinator : Application closed successfully


2025-07-17 14:45:09 : INFO :NAVIGATION_TELLO_WAYPOINT : Navigation to waypoint session closed with drone at current waypoint: WP_002.


2025-07-17 14:45:09 : DEBUG :NAVIGATION_ENGINE : Navigate to waypoint operation completed with drone at current waypoint: WP_002.

(['WP_002'], ['WP_002'])"

can you help me check why is the droe's second movement failing? and the batery poll thread failing? 


 Battery Polling Thread Failure
The battery monitoring thread is failing because it's receiving mixed responses. The battery command is getting responses like 'ok' instead of a battery percentage, causing this error:

2. Second Movement Failure
The second movement is failing because the forward command is timing out and returning battery levels instead of 'ok'. Looking at the logs:

This suggests the drone is responding with battery levels instead of acknowledging the movement command. This happens when:

Commands are sent too quickly without proper delays
The drone is busy processing previous commands
Command queue is mixed up due to concurrent battery polling

The main issues are:

Battery polling conflicts with movement commands
Insufficient delays between commands
No retry logic for failed commands
Command queue confusion due to concurrent operations
"""
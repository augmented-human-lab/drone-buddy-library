"""This is the main python module where we call all the function to do
something...
"""
import sys
import os
import time
from djitellopy import tello

os.environ['QT_QPA_PLATFORM'] = 'xcb'
import cv2

sys.path.insert(0, '/home/zen/drone-buddy-library') 


from dronebuddylib import EngineConfigurations, NavigationAlgorithm, NavigationEngine, AtomicEngineConfigurations
from dronebuddylib.atoms.navigation import NavigationInstruction
from dronebuddylib.utils.logger import Logger

logger = Logger()

def test_navigate_to_waypoint():
    """Test the navigate_to_waypoint function with proper NavigationInstruction enum usage."""
    # Configure navigation engine
    config = EngineConfigurations({})
    config.add_configuration(AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_FILE, 'drone_movements_20250717_143431.json')
    engine = NavigationEngine(NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT, config)
    
    logger.log_info("Main", "Navigation engine initialized successfully")

    # Using proper NavigationInstruction enum values
    # result1 = engine.navigate_to_waypoint("WP_002", NavigationInstruction.CONTINUE)
    # time.sleep(2)  # Wait for the first navigation to complete
    # result2 = engine.navigate_to_waypoint("WP_001", NavigationInstruction.CONTINUE)
    time.sleep(1)  # Wait for the second navigation to complete
    # time.sleep(5)  # Wait for the second navigation to complete
    # result3 = engine.navigate_to_waypoint("WP_001", 1)

    result1 = engine.navigate_to(["WP_002", "WP_003", "Kitchen", "WP_001", "WP_002"], NavigationInstruction.CONTINUE)

    # result2 = engine.navigate_to(["WP_001", "WP_002"], NavigationInstruction.HALT)

    # result = result1.extend(result2)

    return result1

    # return result1, result2

    # return engine.navigate()

    # return engine.map_location()

def main():
    """This is the main function we call when running the python file."""
    
    logger.log_info("Main", "Starting Tello Navigation Tests")
    print(test_navigate_to_waypoint())

if __name__ == "__main__":
    # mav = tello.Tello()
    # mav.connect(wait_for_state=False)
    # mav.streamon()
    # while True:
    #     frame = mav.get_frame_read().frame
    #     cv2.imshow('Tello Video', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()
    # tello.streamoff()
    # tello.end()
    # main()
    # from dronebuddylib.atoms.navigation.tello_waypoint_nav_utils.video_grabber import TelloVideoGrabber
    # tello = Tello()
    # tello.connect(wait_for_state=False)
    # tello.streamon()
    # time.sleep(2)  # Allow time for the video stream to start
    # print("Tello connected successfully")
    # video = TelloVideoGrabber()
    # video.start()
    # try: 
    #     while True:
    #         frame = video.frame
    #         cv2.imshow('Tello Video', frame)
    #         cv2.waitKey(1)
    # except KeyboardInterrupt:
    #     print("Exiting video stream...")
    #     video.stop()
    #     cv2.destroyAllWindows()
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     video.stop()
    #     cv2.destroyAllWindows()
    main()

    """
    (venv) zen@soccf-isa3-001:~/drone-buddy-library$ sudo nmap -sU 192.168.10.1 -p 8889,8890,11111
[sudo] password for zen: 
Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-07-23 18:50 +08
Nmap scan report for 192.168.10.1
Host is up (0.0028s latency).

PORT      STATE         SERVICE
8889/udp  open|filtered ddi-udp-2
8890/udp  closed        ddi-udp-3
11111/udp closed        vce

Nmap done: 1 IP address (1 host up) scanned in 1.45 seconds
    """
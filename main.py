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
    # main()


    # def tello_get_frame(tello: Tello, w=360, h=240):
    #     frame = tello.get_frame_read().frame
    #     if frame is not None:
    #         return cv2.resize(frame, (w, h))
    #     return None
    
    # tello = Tello()
    # tello.connect(wait_for_state=False)
    # tello.streamon()
    # # time.sleep(2)  # Allow time for the video stream to start
    # print("Tello connected successfully")
    # while True:
    #     processed_frame = tello_get_frame(tello, w=360, h=240)
    #     cv2.imshow('Tello Video', processed_frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()
    # tello.streamoff()
    # tello.end()


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

"""
(venv) zen@soccf-isa3-001:~/drone-buddy-library$ python3 main.py
[INFO] tello.py - 129 - Tello instance was initialized. Host: '192.168.10.1'. Port: '8889'.
INFO:djitellopy:Tello instance was initialized. Host: '192.168.10.1'. Port: '8889'.
[INFO] tello.py - 438 - Send command: 'command'
INFO:djitellopy:Send command: 'command'
[ERROR] tello.py - 458 - 'utf-8' codec can't decode byte 0xcc in position 0: invalid continuation byte
ERROR:djitellopy:'utf-8' codec can't decode byte 0xcc in position 0: invalid continuation byte
[INFO] tello.py - 438 - Send command: 'command'
INFO:djitellopy:Send command: 'command'
[INFO] tello.py - 462 - Response command: 'ok'
INFO:djitellopy:Response command: 'ok'
[INFO] tello.py - 438 - Send command: 'streamon'
INFO:djitellopy:Send command: 'streamon'
[INFO] tello.py - 462 - Response streamon: 'ok'
INFO:djitellopy:Response streamon: 'ok'
Tello connected successfully
Traceback (most recent call last):
  File "/home/zen/drone-buddy-library/venv/lib/python3.12/site-packages/djitellopy/tello.py", line 1049, in __init__
    self.container = av.open(self.address, timeout=(Tello.FRAME_GRAB_TIMEOUT, None))
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "av/container/core.pyx", line 420, in av.container.core.open
  File "av/container/core.pyx", line 266, in av.container.core.Container.__cinit__
  File "av/container/core.pyx", line 286, in av.container.core.Container.err_check
  File "av/error.pyx", line 328, in av.error.err_check
av.error.ExitError: [Errno 1414092869] Immediate exit requested: 'udp://@0.0.0.0:11111'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/zen/drone-buddy-library/main.py", line 105, in <module>
    processed_frame = tello_get_frame(tello, w=360, h=240)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zen/drone-buddy-library/main.py", line 94, in tello_get_frame
    frame = tello.get_frame_read().frame
            ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zen/drone-buddy-library/venv/lib/python3.12/site-packages/djitellopy/enforce_types.py", line 54, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/zen/drone-buddy-library/venv/lib/python3.12/site-packages/djitellopy/tello.py", line 421, in get_frame_read
    self.background_frame_read = BackgroundFrameRead(self, address, with_queue, max_queue_len)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zen/drone-buddy-library/venv/lib/python3.12/site-packages/djitellopy/tello.py", line 1051, in __init__
    raise TelloException('Failed to grab video frames from video stream')
djitellopy.tello.TelloException: Failed to grab video frames from video stream
[INFO] tello.py - 438 - Send command: 'streamoff'
INFO:djitellopy:Send command: 'streamoff'
[INFO] tello.py - 462 - Response streamoff: 'ok'
INFO:djitellopy:Response streamoff: 'ok'
"""
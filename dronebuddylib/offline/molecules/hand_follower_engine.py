<<<<<<< HEAD
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from dronebuddylib.offline.atoms.hand_detection import get_hand_landmark
from dronebuddylib.offline.atoms.gesture_recognition import *
=======
from dronebuddylib.offline.atoms.track import *
from dronebuddylib.offline.atoms.hand_detection import get_hand_landmark
from dronebuddylib.offline.atoms.gesture_recognition import *
>>>>>>> origin/main
from djitellopy import Tello
import threading


class HandFollower:
    def __init__(self, tello: Tello):
        self.tello = tello
        self.running = True
<<<<<<< HEAD
        stop_control_thread = threading.Thread(target=self.detect_stop_gesture)
        stop_control_thread.daemon = True
        stop_control_thread.start()
=======
        self.stop_control_thread = threading.Thread(target=self.detect_stop_gesture)
        self.stop_control_thread.daemon = True
        self.stop_control_thread.start()
>>>>>>> origin/main
        self.route = []

    def detect_stop_gesture(self):
        while self.running:
            frame = self.tello.get_frame_read().frame
            landmarks = get_hand_landmark(frame)
<<<<<<< HEAD
            if (is_stop_following(landmarks)):
                self.running = False
=======
            if (landmarks and is_stop_following(landmarks)):
                self.running = False
        self.stop_control_thread.join()
>>>>>>> origin/main


def init_handFollower(tello: Tello):
    """
    Initialize a hand follower.

    Args:
        tello (Tello)

    Returns:
        HandFollower: the initialized hand follower
    """
    return HandFollower(tello)


def fix_target_to_hand(follower: HandFollower):
    """
<<<<<<< HEAD
    Get the drone close the hand in front of it.
=======
    Get the drone close the detected hand.
>>>>>>> origin/main

    Args:
        follower (HandFollower)
    """
    go_count = 0
    while follower.running:
<<<<<<< HEAD
        print(go_count)
        frame = follower.tello.get_frame_read().frame
        result = get_hand_landmark(frame)
        print(result)
=======
        frame = follower.tello.get_frame_read().frame
        result = get_hand_landmark(frame)
>>>>>>> origin/main
        if not result: continue
        tipx = result[8][0]

        rotate = tipx - 0.5
        if rotate < -0.1:
            cmd = "ccw " + str(int(-rotate * 55))
            follower.tello.rotate_counter_clockwise(int(-rotate * 55))
            follower.route.append(cmd)
        if rotate > 0.1:
            cmd = "cw " + str(int(rotate * 55))
            follower.tello.rotate_clockwise(int(rotate * 55))
            follower.route.append(cmd)

        go = (result[5][0] - result[0][0]) * (result[5][0] - result[0][0]) + (result[5][1] - result[0][1]) * (
                result[5][1] - result[0][1])
        if go < 0.01:
            go_count = go_count + 1
<<<<<<< HEAD
        elif go > 0.04:
=======
        elif go > 0.045:
>>>>>>> origin/main
            go_count = go_count - 1
        else:
            go_count = 0
        if go_count >= 2:
            cmd = "forward " + str(int((0.02 - go) * 2000))
            follower.tello.move_forward(int((0.02 - go) * 2000))
            follower.route.append(cmd)
            go_count = 0
        if go_count <= -2:
            cmd = "back " + str(int((go - 0.02) * 800))
            follower.tello.move_back(int((go - 0.02) * 800))
            follower.route.append(cmd)
            go_count = 0

        h = result[5][1] - 0.5
        if h < -0.25:
            cmd = "up " + str(30)
            follower.tello.move_up(30)
            follower.route.append(cmd)
        if h > 0.25:
            cmd = "down " + str(30)
            follower.tello.move_down(30)
            follower.route.append(cmd)
<<<<<<< HEAD
        
=======
    return  
>>>>>>> origin/main

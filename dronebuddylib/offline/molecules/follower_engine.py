<<<<<<< HEAD
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from dronebuddylib.offline.atoms.track import *
from dronebuddylib.offline.atoms.hand_detection import get_hand_landmark
from dronebuddylib.offline.atoms.gesture_recognition import is_stop_following
=======
from dronebuddylib.offline.atoms.track import *
from dronebuddylib.offline.atoms.hand_detection import get_hand_landmark
from dronebuddylib.offline.atoms.gesture_recognition import is_stop_following
>>>>>>> origin/main
from djitellopy import Tello
import threading


class Follower:
    def __init__(self, tracker: TrackEngine, tello: Tello):
        self.tello = tello
        self.tracker = tracker
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


def init_follower(tracker: TrackEngine, tello: Tello):
    """
    Initialize a follower.

    Args:
        tracker (TrackEngine): an initialized follower with a target set 
        tello (Tello)

    Returns:
        Follower: an initialized follower
    """
    return Follower(tracker, tello)


def follow(follower: Follower):
    """
    Args:
        follower (Follower): intialized follower
    """
    go_count = 0
    while follower.running:
        frame = follower.tello.get_frame_read().frame
        xywh = get_tracked_bounding_box(frame, follower.tracker)
        sizey, sizex, _ = frame.shape
<<<<<<< HEAD
        print("result:   ", xywh)
=======
>>>>>>> origin/main
        follower.tello.rotate_clockwise(1)
        follower.tello.rotate_counter_clockwise(1)
        if not xywh == False:
            x = xywh[0]
            y = xywh[1]
            w = xywh[2]
            h = xywh[3]
            x = x + w / 2
            y = y + h / 2
            x = x / sizex
            y = y / sizey
            w = w / sizex
            h = h / sizey

            rotate = x - 0.5
            if rotate < -0.2:
<<<<<<< HEAD
                print(1)
                follower.tello.rotate_counter_clockwise(int(-rotate * 60))
            if rotate > 0.2:
                print(2)
=======
                follower.tello.rotate_counter_clockwise(int(-rotate * 60))
            if rotate > 0.2:
>>>>>>> origin/main
                follower.tello.rotate_clockwise(int(rotate * 60))

            go = w - 0.3
            if go < -0.1:
                go_count = go_count + 1
            elif go > 0.1:
                go_count = go_count - 1
            else:
                go_count = 0
            if go_count >= 2:
<<<<<<< HEAD
                print(3)
                follower.tello.move_forward(int(-go * 350))
                go_count = 0
            if go_count <= -2:
                print(4)
=======
                follower.tello.move_forward(int(-go * 350))
                go_count = 0
            if go_count <= -2:
>>>>>>> origin/main
                follower.tello.move_back(int(go * 350))
                go_count = 0

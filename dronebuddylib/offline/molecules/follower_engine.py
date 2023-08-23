import threading

from djitellopy import Tello

from dronebuddylib.offline.atoms.gesture_recognition import is_stop_following
from dronebuddylib.offline.atoms.hand_detection import get_hand_landmark
from dronebuddylib.offline.atoms.basic_tracking import *


class Follower:
    """
      A class representing a drone follower behavior using hand gestures.

      Args:
          tracker (TrackEngine): The object tracker instance for tracking an object.
          tello (Tello): The Tello drone instance.

      Attributes:
          tello (Tello): The Tello drone instance.
          tracker (TrackEngine): The object tracker instance for tracking an object.
          running (bool): Flag indicating if the follower behavior is active.
          stop_control_thread (threading.Thread): Thread for detecting the stop gesture.

      Example:
          tello_instance = Tello()
          tracker_instance = TrackEngine()
          follower = Follower(tracker_instance, tello_instance)
          follower.detect_stop_gesture()
      """
    def __init__(self, tracker: TrackEngine, tello: Tello):
        """
              Initializes the Follower instance.

              Args:
                  tracker (TrackEngine): The object tracker instance for tracking an object.
                  tello (Tello): The Tello drone instance.
       """
        self.tello = tello
        self.tracker = tracker
        self.running = True
        self.stop_control_thread = threading.Thread(target=self.detect_stop_gesture)
        self.stop_control_thread.daemon = True
        self.stop_control_thread.start()

    def detect_stop_gesture(self):
        """
               Monitors the camera feed for a stop gesture and terminates the follower behavior if detected.
        """
        while self.running:
            frame = self.tello.get_frame_read().frame
            landmarks = get_hand_landmark(frame)
            if landmarks and is_stop_following(landmarks):
                self.running = False
        self.stop_control_thread.join()


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
                follower.tello.rotate_counter_clockwise(int(-rotate * 60))
            if rotate > 0.2:
                follower.tello.rotate_clockwise(int(rotate * 60))

            go = w - 0.3
            if go < -0.1:
                go_count = go_count + 1
            elif go > 0.1:
                go_count = go_count - 1
            else:
                go_count = 0
            if go_count >= 2:
                follower.tello.move_forward(int(-go * 350))
                go_count = 0
            if go_count <= -2:
                follower.tello.move_back(int(go * 350))
                go_count = 0

from djitellopy import Tello

from dronebuddylib.offline.atoms.head_bounding import get_head_bounding_box
from .follower_engine import *


def follow_person(tello: Tello, path: str):
    """
    Follow the person in front of the drone.

    Args:
        tello (Tello)
        path (str): the absolute path to directory of the two .pth files for the tracker
    """
    frame, box = get_head_bounding_box(tello)

    tracker_engine = TrackingEngine()
    tracker_engine.init_tracker(path)
    tracker_engine.set_target(frame, box)

    follower = Follower(tracker_engine, tello)
    follower.follow_person_with_gesture()
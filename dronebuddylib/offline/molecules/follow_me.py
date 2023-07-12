from .follow import *
from djitellopy import Tello
from dronebuddylib.offline.atoms.head_bounding import get_head_bounding_box
from dronebuddylib.offline.atoms.track import *

def follow_me(tello: Tello, path: str):
  """
  Follow the person in front of the drone.

  Args:
      tello (Tello)
      path (str): the absolute path to directory of the two .pth files for the tracker
  """
  frame, box = get_head_bounding_box(tello)
  tracker = init_tracker(path)
  set_target(frame, box, tracker)
  follower = init_follower(tracker, tello)
  follow(follower)
  
  
  
from follow import *
from djitellopy import Tello
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from atoms.head_bounding import get_head_bounding_box
from atoms.track import *

def follow_me(tello: Tello):
  """
  Follow the person in front of the drone.

  Args:
      tello (Tello)
  """
  frame, box = get_head_bounding_box(tello)
  tracker = init_tracker()
  set_target(frame, box, tracker)
  follower = init_follower(tracker, tello)
  follow(follower)
  
  
  
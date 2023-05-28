from follow import *
from djitellopy import Tello
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from atoms.head_bounding import head_bounding_box
from atoms.track import *

def follow_me(tello: Tello):
  """
  Follow the person in front of the drone.

  Args:
      tello (Tello)
  """
  frame, box = head_bounding_box(tello)
  tracker = TrackEngine()
  init(frame, box, tracker)
  follower = Follower(tracker, tello)
  follow(follower)
  
  
  
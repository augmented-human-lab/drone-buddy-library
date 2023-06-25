import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent) + '\\resources\\tracking')
from resources.tracking.pytracking.evaluation import Tracker
import cv2

class TrackEngine:
  def __init__(self):
    tracker_name = 'tomp'
    tracker_params = 'tomp101'
    tracker_creater = Tracker(tracker_name, tracker_params)
    params = tracker_creater.get_parameters()
    debug_ = getattr(params, 'debug', 0)
    params.debug = debug_
    params.tracker_name = tracker_name
    params.param_name = tracker_params
    tracker_creater._init_visdom(None, debug_)
    self.tracker = tracker_creater.create_tracker(params)

def _build_init_info(box):
  return {'init_bbox': box, 'init_object_ids': [1, ], 'object_ids': [1, ],
          'sequence_object_ids': [1, ]}

def init_tracker():
  """
  Initialize a tracker

  Returns:
      TrackEngine: the initialized tracker.
  """
  return TrackEngine()

def set_target(img: list, box: list, tracker: TrackEngine):
  """
  Set the TrackEngine to track certain object.

  Args:
      img (list): the current frame
      box (list): the bounding box of the object to be tracked
      tracker (TrackEngine)
  """
  tracker.tracker.initialize(img, _build_init_info(box))

def get_tracked_bounding_box(img: list, tracker: TrackEngine):
  """
  Locate the object in an image.

  Args:
      img (list): the current frame
      tracker (TrackEngine): the intialized tracker

  Returns:
      list | bool: the bounding box of the tracked object in this frame.
      Return False if no target is found.
  """
  frame = img.copy()
  out = tracker.tracker.track(frame)
  state = [int(s) for s in out['target_bbox']]
  score = out['object_presence_score']
  cv2.rectangle(frame, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
            (0, 255, 0), 5)
  cv2.imshow(" ", frame)
  cv2.waitKey(1)
  if score < 0.7:
      return False
  return state
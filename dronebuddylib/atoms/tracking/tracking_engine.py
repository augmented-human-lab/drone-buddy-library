import os
import shutil
import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent) + '\\resources\\tracking')
from dronebuddylib.atoms.resources.tracking.pytracking.evaluation import Tracker


class TrackingEngine:
    """
     Description of the TrackEngine class.
     """

    def __init__(self):
        """
               Description of the constructor.
               """

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

    def _build_init_info(self, box):
        """
               Description of the constructor.
               """
        return {'init_bbox': box, 'init_object_ids': [1, ], 'object_ids': [1, ],
                'sequence_object_ids': [1, ]}

    def init_tracker(self, path: str):
        """
        Initialize a tracker

        Args:
              path (str): the absolute path to directory of the two .pth files for the tracker

        Returns:
            TrackEngine: the initialized tracker.
        """
        destination_dir = str(Path(__file__).resolve().parent) + '\\resources\\tracking\\pytracking\\networks'
        source_path = os.path.join(path, 'keep_track.pth.tar')
        destination_path = os.path.join(destination_dir, 'keep_track.pth.tar')
        shutil.copy2(source_path, destination_path)
        source_path = os.path.join(path, 'tomp101.pth.tar')
        destination_path = os.path.join(destination_dir, 'tomp101.pth.tar')
        shutil.copy2(source_path, destination_path)

    def set_target(self, image_frame: list, box: list):
        """
        Set the TrackEngine to track certain object.

        Args:
            image_frame (list): the current frame
            box (list): the bounding box of the object to be tracked
            tracker (TrackEngine)
        """
        self.tracker.initialize(image_frame, self._build_init_info(box))

    def get_tracked_bounding_box(self, image_frame: list):
        """
        Locate the object in an image.

        Args:
            image_frame (list): the current frame
            tracker (TrackEngine): the intialized tracker

        Returns:
            list | bool: the bounding box of the tracked object in this frame.
            Return False if no target is found.
        """
        frame = image_frame.copy()
        out = self.tracker.track(frame)
        state = [int(s) for s in out['target_bbox']]
        score = out['object_presence_score']
        cv2.rectangle(frame, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                      (0, 255, 0), 5)
        cv2.imshow(" ", frame)
        cv2.waitKey(1)
        if score < 0.7:
            return False
        return state

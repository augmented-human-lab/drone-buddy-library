import threading

from djitellopy import Tello

from dronebuddylib.atoms.gesture_recognition import is_stop_following, is_pointing
from dronebuddylib.atoms.hand_detection import get_hand_landmark
from offline.atoms import get_bounding_boxes, select_pointed_obj
from offline.atoms.objectdetection.vision_configs import VisionConfigs
from offline.atoms.tracking.tracking_engine import TrackingEngine
from utils.enums import VisionAlgorithm


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

    def __init__(self, tracker: TrackingEngine, tello: Tello):
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

    def follow_person_with_gesture(self):

        go_count = 0
        while self.running:
            frame = self.tello.get_frame_read().frame
            xywh = self.tracker.get_tracked_bounding_box(frame)
            sizey, sizex, _ = frame.shape
            self.tello.rotate_clockwise(1)
            self.tello.rotate_counter_clockwise(1)
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
                    self.tello.rotate_counter_clockwise(int(-rotate * 60))
                if rotate > 0.2:
                    self.tello.rotate_clockwise(int(rotate * 60))

                go = w - 0.3
                if go < -0.1:
                    go_count = go_count + 1
                elif go > 0.1:
                    go_count = go_count - 1
                else:
                    go_count = 0
                if go_count >= 2:
                    self.tello.move_forward(int(-go * 350))
                    go_count = 0
                if go_count <= -2:
                    self.tello.move_back(int(go * 350))
                    go_count = 0

    def get_pointed_obj(self, algorithm: VisionAlgorithm, vision_config: VisionConfigs):
        """Get the bounding box of the pointed object and the frame once the drone detects a pointing hand gesture.

        Args:
            follower (HandFollower): an initialized follower
            algorithm (VisionAlgorithm): the algorithm used for object detection
            vision_config (VisionConfigs): the configuration for the object detection algorithm

        Returns:
            tuple[list, list] | None: the frame including the pointing gesture, the bounding box of the pointed object
        """
        go_count = 0
        while self.running:
            frame = self.tello.get_frame_read().frame
            result = get_hand_landmark(frame)
            if not result: continue
            if is_pointing(result):
                obj_boxes = get_bounding_boxes(algorithm, vision_config, frame)
                pointed_obj = select_pointed_obj(frame, result, obj_boxes)
                if pointed_obj:
                    return frame, pointed_obj

            tipx = result[8][0]

            rotate = tipx - 0.5  # Deviation to the middle
            # The following threshold values and coefficients are derived by real test,
            # which could be scaled up and down based on different size of the room
            if rotate < -0.1:
                cmd = "ccw " + str(
                    int(-rotate * 55))  # The Angle of rotation is proportional to the degree of deviation
                self.tello.rotate_counter_clockwise(int(-rotate * 55))
                self.route.append(cmd)
            if rotate > 0.1:
                cmd = "cw " + str(int(rotate * 55))
                self.tello.rotate_clockwise(int(rotate * 55))
                self.route.append(cmd)

            go = (result[5][0] - result[0][0]) * (result[5][0] - result[0][0]) + (result[5][1] - result[0][1]) * (
                    result[5][1] - result[0][1])  # The palm length
            if go < 0.01:
                go_count = go_count + 1
            elif go > 0.04:
                go_count = go_count - 1
            else:
                go_count = 0
            if go_count >= 2:  # move forward or backward if the palm is too big or small in two consecutive frames
                cmd = "forward " + str(int((0.02 - go) * 2000))
                self.tello.move_forward(int((0.02 - go) * 2000))
                self.route.append(cmd)
                go_count = 0
            if go_count <= -2:
                cmd = "back " + str(int((go - 0.02) * 800))
                self.tello.move_back(int((go - 0.02) * 800))
                self.route.append(cmd)
                go_count = 0

            h = result[5][1] - 0.5
            if h < -0.25:
                cmd = "up " + str(30)
                self.tello.move_up(30)
                self.route.append(cmd)
            if h > 0.25:
                cmd = "down " + str(30)
                self.tello.move_down(30)
                self.route.append(cmd)

        return [], []

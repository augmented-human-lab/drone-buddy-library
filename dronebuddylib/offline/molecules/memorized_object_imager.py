import os
import shutil
import time

import cv2
from djitellopy import Tello

from offline.atoms.tracking.tracking_engine import TrackingEngine

"""
This is the docstring for the dronebuddylib.online.molecules module.
"""


class MemorizedObjectImager:
    """
      A class representing a drone's flight behavior for capturing images of memorized objects.

      Args:
          tello (Tello): The Tello drone instance.
          name (str): The name of the memorized object.
          tracker_engine (TrackingEngine): The object tracker instance for tracking the memorized object.

      Attributes:
          tello (Tello): The Tello drone instance.
          tracker (TrackingEngine): The object tracker instance for tracking the memorized object.
          img_dir (str): The directory to store captured images of the memorized object.

      Example:
          tello_instance = Tello()
          tracker_instance = TrackingEngine()
          memorized_object_imager = MemorizedObjectImager(tello_instance, "my_object", tracker_instance)
          MemorizedObjectImager.cut("image_1.jpg")
      """

    def __init__(self, tello: Tello, name: str, tracker_engine: TrackingEngine):
        """
                Initializes the FlyArrounder instance.

                Args:
                    tello (Tello): The Tello drone instance.
                    name (str): The name of the memorized object.
                    tracker (Tracker): The object tracker instance for tracking the memorized object.
                """
        self.tello = tello
        self.tracker = tracker_engine
        self.img_dir = str(Path(__file__).resolve().parent) + '\\memorized_obj_photo\\' + name + "\\"
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)
        else:
            shutil.rmtree(self.img_dir)
            os.mkdir(self.img_dir)

    def cut(self, img_name: str):
        """
               Captures an image of the tracked memorized object from the drone's camera feed.

               Args:
                   img_name (str): The name to be given to the captured image.
        """
        frame = self.tello.get_frame_read().frame
        xywh = self.tracker.get_tracked_bounding_box(frame)
        if (xywh == False):
            return
        min_row = xywh[1]
        max_row = xywh[1] + xywh[3]
        min_col = xywh[0]
        max_col = xywh[0] + xywh[2]
        cut_img = frame[min_row:max_row, min_col:max_col]
        cv2.imwrite(self.img_dir + img_name, cut_img)

    def capture_images_around_object(self, frame, box):
        """Fly around and take photos for the object for memorizing

        Args:
            frame (list): the current frame
            box (list): the bounding box around the object in this frame
        """
        self.tracker.set_target(frame, box)
        time.sleep(2)
        # The following angles and distances are derived by real tests
        # allowing the drone to fly around the object and take photos from different positions
        self.cut("0.jpg")
        self.tello.rotate_counter_clockwise(15)
        self.tello.move_right(30)
        self.cut("1.jpg")
        self.tello.rotate_counter_clockwise(15)
        self.tello.move_right(30)
        self.cut("2.jpg")
        self.tello.move_left(30)
        self.tello.rotate_clockwise(15)
        self.tello.move_left(30)
        self.tello.rotate_clockwise(30)
        self.tello.move_left(30)
        self.cut("3.jpg")
        self.tello.rotate_clockwise(15)
        self.tello.move_left(30)
        self.cut("4.jpg")
        self.tello.move_right(30)
        self.tello.move_forward(20)
        self.cut("8.jpg")
        self.tello.move_back(50)
        self.cut("5.jpg")
        self.tello.move_up(30)
        self.cut("6.jpg")
        self.tello.move_down(50)
        self.cut("7.jpg")

        return

import os
import shutil
import time

import cv2
from djitellopy import Tello

from dronebuddylib.offline.atoms.basic_tracking import *

"""
This is the docstring for the dronebuddylib.online.molecules module.
"""
class FlyArrounder:
    """
      A class representing a drone's flight behavior for capturing images of memorized objects.

      Args:
          tello (Tello): The Tello drone instance.
          name (str): The name of the memorized object.
          tracker (Tracker): The object tracker instance for tracking the memorized object.

      Attributes:
          tello (Tello): The Tello drone instance.
          tracker (Tracker): The object tracker instance for tracking the memorized object.
          img_dir (str): The directory to store captured images of the memorized object.

      Example:
          tello_instance = Tello()
          tracker_instance = Tracker()
          fly_arrounder = FlyArrounder(tello_instance, "my_object", tracker_instance)
          fly_arrounder.cut("image_1.jpg")
      """
    def __init__(self, tello: Tello, name: str, tracker: Tracker):
        """
                Initializes the FlyArrounder instance.

                Args:
                    tello (Tello): The Tello drone instance.
                    name (str): The name of the memorized object.
                    tracker (Tracker): The object tracker instance for tracking the memorized object.
                """
        self.tello = tello
        self.tracker = tracker
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
        xywh = get_tracked_bounding_box(frame, self.tracker)
        if (xywh == False):
            return
        min_row = xywh[1]
        max_row = xywh[1] + xywh[3]
        min_col = xywh[0]
        max_col = xywh[0] + xywh[2]
        cut_img = frame[min_row:max_row, min_col:max_col]
        cv2.imwrite(self.img_dir + img_name, cut_img)


def init_fly_arrounder(tello: Tello, name: str, tracker: Tracker):
    """initiate an engine for flying around the object.

    Args:
        tello (Tello): 
        name (str): the label for the object
        tracker (Tracker): an initialized tracker

    Returns:
        FlyArrounder: the initialized FlyArounder engine
    """
    return FlyArrounder(tello, name, tracker)


def fly_around(fly_arrounder: FlyArrounder, frame, box):
    """Fly around and take photos for the object for memorizing

    Args:
        fly_arrounder (FlyArrounder): the initialized FlyArounder engine
        frame (list): the current frame
        box (list): the bounding box around the object in this frame
    """
    set_target(frame, box, fly_arrounder.tracker)
    time.sleep(2)
    # The following angles and distances are derived by real tests 
    # allowing the drone to fly around the object and take photos from different positions
    fly_arrounder.cut("0.jpg")
    fly_arrounder.tello.rotate_counter_clockwise(15)
    fly_arrounder.tello.move_right(30)
    fly_arrounder.cut("1.jpg")
    fly_arrounder.tello.rotate_counter_clockwise(15)
    fly_arrounder.tello.move_right(30)
    fly_arrounder.cut("2.jpg")
    fly_arrounder.tello.move_left(30)
    fly_arrounder.tello.rotate_clockwise(15)
    fly_arrounder.tello.move_left(30)
    fly_arrounder.tello.rotate_clockwise(30)
    fly_arrounder.tello.move_left(30)
    fly_arrounder.cut("3.jpg")
    fly_arrounder.tello.rotate_clockwise(15)
    fly_arrounder.tello.move_left(30)
    fly_arrounder.cut("4.jpg")
    fly_arrounder.tello.move_right(30)
    fly_arrounder.tello.move_forward(20)
    fly_arrounder.cut("8.jpg")
    fly_arrounder.tello.move_back(50)
    fly_arrounder.cut("5.jpg")
    fly_arrounder.tello.move_up(30)
    fly_arrounder.cut("6.jpg")
    fly_arrounder.tello.move_down(50)
    fly_arrounder.cut("7.jpg")
    return

from dronebuddylib.offline.atoms.track import *
from djitellopy import Tello
import os
import shutil
import cv2
import time

class FlyArounder:
    def __init__(self, tello: Tello, name: str, tracker: Tracker):
        self.tello = tello
        self.tracker = tracker
        self.img_dir = str(Path(__file__).resolve().parent) + '\\memorized_obj_photo\\' + name + "\\"
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)
        else:
            shutil.rmtree(self.img_dir)
            os.mkdir(self.img_dir)
    
    def cut(self, img_name: str):
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

def init_flyArounder(tello: Tello, name: str, tracker: Tracker):
    """initiate an engine for flying around the object.

    Args:
        tello (Tello): 
        name (str): the label for the object
        tracker (Tracker): an initialized tracker

    Returns:
        FlyArounder: the initialized FlyArounder engine
    """
    return FlyArounder(tello, name, tracker)

def fly_around(flyArounder: FlyArounder, frame, box):
    """Fly around and take photos for the object for memorizing

    Args:
        flyArounder (FlyArounder): the initialized FlyArounder engine
        frame (list): the current frame
        box (list): the bounding box around the object in this frame
    """
    set_target(frame, box, flyArounder.tracker)
    time.sleep(2)
    # The following angles and distances are derived by real tests 
    # allowing the drone to fly around the object and take photos from different positions
    flyArounder.cut("0.jpg")
    flyArounder.tello.rotate_counter_clockwise(15)
    flyArounder.tello.move_right(30)
    flyArounder.cut("1.jpg")
    flyArounder.tello.rotate_counter_clockwise(15)
    flyArounder.tello.move_right(30)
    flyArounder.cut("2.jpg")
    flyArounder.tello.move_left(30)
    flyArounder.tello.rotate_clockwise(15)
    flyArounder.tello.move_left(30)
    flyArounder.tello.rotate_clockwise(30)
    flyArounder.tello.move_left(30)
    flyArounder.cut("3.jpg")
    flyArounder.tello.rotate_clockwise(15)
    flyArounder.tello.move_left(30)
    flyArounder.cut("4.jpg")
    flyArounder.tello.move_right(30)
    flyArounder.tello.move_forward(20)
    flyArounder.cut("8.jpg")
    flyArounder.tello.move_back(50)
    flyArounder.cut("5.jpg")
    flyArounder.tello.move_up(30)
    flyArounder.cut("6.jpg")
    flyArounder.tello.move_down(50)
    flyArounder.cut("7.jpg")
    return
        
        
    
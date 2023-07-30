import unittest

import time
import dronebuddylib.offline.atoms as dbl_atoms
import dronebuddylib.offline.molecules as dbl_molecules
from djitellopy import Tello

# This test could test tracking, object_detection_yolo, object_selection, get_pointed_obj, 
# hand_following, hand_detection and gesture_recognition
class TestMemorize(unittest.TestCase):

    def test_memorize(self):
        try: 
            tello = Tello()
            tello.connect()
            tello.streamon()
            tello.set_video_resolution(Tello.RESOLUTION_720P)
            tello.get_frame_read().frame
            time.sleep(4)
            tello.takeoff()
            tello.move_up(80)
            yoloEngine = dbl_atoms.init_yolo_engine(r"C:\Users\wangz\drone\yolov3.weights")
            handFollower = dbl_molecules.init_handFollower(tello)
            frame, bounding_box = dbl_molecules.get_pointed_obj(handFollower, yoloEngine)
            print(bounding_box)
            if frame == []:
                tello.land()
                return
            tracker = dbl_atoms.init_tracker(r"C:\Users\wangz\drone")
            flyArounder = dbl_molecules.init_flyArounder(tello, "cup", tracker)
            dbl_molecules.fly_around(flyArounder, frame, bounding_box)
            tello.land()
            tello.streamoff()
            dbl_atoms.update_memory()
        except Exception as e:
            print(e)


if __name__ == '__main__':
    unittest.main()

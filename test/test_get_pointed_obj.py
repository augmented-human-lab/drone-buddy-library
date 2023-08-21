import time
import unittest

from djitellopy import Tello

import offline as dbl


# This test could test tracking, object_detection_yolo, object_selection, get_pointed_obj, hand_following, hand_detection and gesture_recognition
class TestGetPointedObj(unittest.TestCase):

    def test_get_pointed_obj(self):
        tello = Tello()
        tello.connect()
        tello.streamon()
        tello.get_frame_read().frame
        time.sleep(4)
        tello.takeoff()
        tello.move_up(70)
        yoloEngine = dbl.atoms.init_yolo_engine("PATH_TO_WEIGHTS")
        handFollower = dbl.molecules.init_handFollower(tello)
        frame, bounding_box = dbl.molecules.get_pointed_obj(handFollower, yoloEngine)
        print(bounding_box)


if __name__ == '__main__':
    unittest.main()

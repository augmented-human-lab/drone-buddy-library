import unittest

import time
import offline.molecules as dbl
from djitellopy import Tello

# This test could test follow, follow_me, tracking, get_head_bounding_box and face_detection
class TestFollowMe(unittest.TestCase):

    def test_follow_me(self):
        tello = Tello()
        tello.connect()
        tello.streamon()
        tello.get_frame_read().frame
        time.sleep(4)
        tello.takeoff()
        tello.move_up(70)
        dbl.follow_me(tello)


if __name__ == '__main__':
    unittest.main()

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from atoms.track import *
from atoms.hand_detection import get_hand_landmark
from atoms.gesture_recognition import *
from atoms.object_detection import detect_objects
from atoms.select_pointed_obj import select_pointed_obj
from djitellopy import Tello
from close_to_hand import Hand_follower
import threading

def get_pointed_obj(follower: Hand_follower):
    """
    Get the bounding box of the pointed object and the frame once the drone detects a pointing hand gesture.

    Args:
        follower (Hand_follower): an initialized follower

    Returns:
        list, list: the frame including the pointing gesture, the bounding box of the pointed object
    """
    go_count = 0
    while follower.running:
        print(go_count)
        frame = follower.tello.get_frame_read().frame
        result = get_hand_landmark(frame)
        print(result)
        if (result == False): continue
        if (is_pointing(result)):
            obj_boxes = detect_objects(frame)
            pointed_obj = select_pointed_obj(frame, result, obj_boxes)
            if(pointed_obj):
                # cv2.rectangle(frame, (pointed_obj[0], pointed_obj[1]), (pointed_obj[0] + pointed_obj[2], pointed_obj[1] + pointed_obj[3]), (0, 255, 0), 3)
                # cv2.imshow("123", frame)
                # cv2.waitKey(0)
                return frame, pointed_obj
        
        tipx = result[8][0]

        rotate = tipx - 0.5
        if rotate < -0.1:
            cmd = "ccw " + str(int(-rotate * 55))
            follower.tello.rotate_counter_clockwise(int(-rotate * 55))
            follower.route.append(cmd)
        if rotate > 0.1:
            cmd = "cw " + str(int(rotate * 55))
            follower.tello.rotate_clockwise(int(rotate * 55))
            follower.route.append(cmd)
        

        go = (result[5][0] - result[0][0]) * (result[5][0] - result[0][0]) + (result[5][1] - result[0][1]) * (result[5][1] - result[0][1])
        if go < 0.01:
            go_count = go_count + 1
        elif go > 0.04:
            go_count = go_count - 1
        else:
            go_count = 0
        if go_count >= 2:
            cmd = "forward " + str(int((0.02 - go) * 2000))
            follower.tello.move_forward(int((0.02 - go) * 2000))
            follower.route.append(cmd)
            go_count = 0
        if go_count <= -2:
            cmd = "back " + str(int((go - 0.02) * 800))
            follower.tello.move_back(int((go - 0.02) * 800))
            follower.route.append(cmd)
            go_count = 0

        h = result[5][1] - 0.5
        if h < -0.25:
            cmd = "up " + str(30)
            follower.tello.move_up(30)
            follower.route.append(cmd)
        if h > 0.25:
            cmd = "down " + str(30)
            follower.tello.move_down(30)
            follower.route.append(cmd)
        
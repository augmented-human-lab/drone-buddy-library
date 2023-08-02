from dronebuddylib.offline.atoms.track import *
from dronebuddylib.offline.atoms.hand_detection import get_hand_landmark
from dronebuddylib.offline.atoms.gesture_recognition import *
from dronebuddylib.offline.atoms.object_detection_yolo import *
from dronebuddylib.offline.atoms.object_selection import select_pointed_obj
from .hand_following import HandFollower

def get_pointed_obj(follower: HandFollower, yoloEngine: YoloEngine):
    """Get the bounding box of the pointed object and the frame once the drone detects a pointing hand gesture.

    Args:
        follower (HandFollower): an initialized follower
        yoloEngine (YoloEngine): an initialized YoloEngine

    Returns:
        tuple[list, list] | None: the frame including the pointing gesture, the bounding box of the pointed object
    """
    go_count = 0
    while follower.running:
        frame = follower.tello.get_frame_read().frame
        result = get_hand_landmark(frame)
        if (result == False): continue
        if (is_pointing(result)):
            obj_boxes = get_boxes_yolo(yoloEngine, frame)
            pointed_obj = select_pointed_obj(frame, result, obj_boxes)
            if(pointed_obj):
                return frame, pointed_obj
        
        tipx = result[8][0]

        rotate = tipx - 0.5 # Deviation to the middle
        # The following threshold values and coefficients are derived by real test,
        # which could be scaled up and down based on different size of the room
        if rotate < -0.1:
            cmd = "ccw " + str(int(-rotate * 55)) # The Angle of rotation is proportional to the degree of deviation
            follower.tello.rotate_counter_clockwise(int(-rotate * 55))
            follower.route.append(cmd)
        if rotate > 0.1:
            cmd = "cw " + str(int(rotate * 55))
            follower.tello.rotate_clockwise(int(rotate * 55))
            follower.route.append(cmd)
        
        go = (result[5][0] - result[0][0]) * (result[5][0] - result[0][0]) + (result[5][1] - result[0][1]) * (result[5][1] - result[0][1]) # The palm length 
        if go < 0.01:
            go_count = go_count + 1
        elif go > 0.04:
            go_count = go_count - 1
        else:
            go_count = 0
        if go_count >= 2: # move forward or backward if the palm is too big or small in two consecutive frames
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

    return [], []
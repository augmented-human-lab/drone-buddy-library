from dronebuddylib.offline.atoms.track import *
from dronebuddylib.offline.atoms.hand_detection import get_hand_landmark
from dronebuddylib.offline.atoms.gesture_recognition import *
from djitellopy import Tello
import threading
    
class HandFollower:
    def __init__(self, tello: Tello):
        self.tello = tello
        self.running = True
        self.stop_control_thread = threading.Thread(target=self.detect_stop_gesture)
        self.stop_control_thread.daemon = True
        self.stop_control_thread.start()
        self.route = []
    
    def detect_stop_gesture(self):
        while self.running:
            frame = self.tello.get_frame_read().frame
            landmarks = get_hand_landmark(frame)
            if (landmarks and is_stop_following(landmarks)):
                self.running = False
        self.stop_control_thread.join()

def init_handFollower(tello: Tello):
    """
    Initialize a hand follower.

    Args:
        tello (Tello)

    Returns:
        HandFollower: the initialized hand follower
    """
    return HandFollower(tello)

def close_to_hand(follower: HandFollower):
    """
    Get the drone close the detected hand.

    Args:
        follower (HandFollower)
    """
    go_count = 0
    while follower.running:
        frame = follower.tello.get_frame_read().frame
        result = get_hand_landmark(frame)
        if (result == False): continue
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
        elif go > 0.045:
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
    return  
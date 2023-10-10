import threading

from djitellopy import Tello

from dronebuddylib.atoms.gesture_recognition import *
from dronebuddylib.atoms.hand_detection import get_hand_landmark


class HandFollower:
    """
        A class representing a drone follower behavior using hand gestures.

        Args:
            tello (Tello): The Tello drone instance.

        Attributes:
            tello (Tello): The Tello drone instance.
            running (bool): Flag indicating if the follower behavior is active.
            stop_control_thread (threading.Thread): Thread for detecting the stop gesture.
            route (list): List to store the route followed by the drone.

        Example:
            tello_instance = Tello()
            hand_follower = HandFollower(tello_instance)
            hand_follower.detect_stop_gesture()
        """

    def __init__(self, tello: Tello):
        """
        Initializes the HandFollower instance.

        Args:
            tello (Tello): The Tello drone instance.
        """
        self.tello = tello
        self.running = True
        self.stop_control_thread = threading.Thread(target=self.detect_stop_gesture)
        self.stop_control_thread.daemon = True
        self.stop_control_thread.start()
        self.route = []

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

    def close_to_hand(self):
        """
        Get the drone close the detected hand.

        Args:
            follower (HandFollower)
        """
        go_count = 0
        while self.running:
            frame = self.tello.get_frame_read().frame
            result = get_hand_landmark(frame)
            if result == False: continue
            tipx = result[8][0]

            rotate = tipx - 0.5
            if rotate < -0.1:
                cmd = "ccw " + str(int(-rotate * 55))
                self.tello.rotate_counter_clockwise(int(-rotate * 55))
                self.route.append(cmd)
            if rotate > 0.1:
                cmd = "cw " + str(int(rotate * 55))
                self.tello.rotate_clockwise(int(rotate * 55))
                self.route.append(cmd)

            go = (result[5][0] - result[0][0]) * (result[5][0] - result[0][0]) + (result[5][1] - result[0][1]) * (
                    result[5][1] - result[0][1])
            if go < 0.01:
                go_count = go_count + 1
            elif go > 0.045:
                go_count = go_count - 1
            else:
                go_count = 0
            if go_count >= 2:
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
        return

import os
import re
import threading
import time
import unittest
from collections import Counter

import cv2

from djitellopy import Tello

from dronebuddylib.atoms.placerecognition import PlaceRecognitionKNNImpl
from dronebuddylib.atoms.placerecognition.place_recognition_engine import PlaceRecognitionEngine
from dronebuddylib.models.engine_configurations import EngineConfigurations, logger
from dronebuddylib.models.enums import FaceRecognitionAlgorithm, AtomicEngineConfigurations, PlaceRecognitionAlgorithm

is_drone_in_air = False

import socketio

# Create a Socket.IO client
sio = socketio.Client()


@sio.event
def connect():
    print("I'm connected to the server.")
    # Send a voice request message to the server


def connect_to_voice_server():
    sio.connect('http://127.0.0.1:65432')


def say_out_loud(text):
    sio.emit('voice_request', text)


def test_face_rec_knn_recognize_place_by_video():
    global is_drone_in_air
    tello = Tello()
    tello.connect()
    tello.streamon()
    tello.get_frame_read().frame
    tello.takeoff()
    is_drone_in_air = True
    print("battery: ", tello.get_battery())
    print("temperature: ", tello.get_temperature())

    thread3 = threading.Thread(target=keep_drone_in_air, args=(tello,))
    thread_recognize = threading.Thread(target=test_determine_place_on_video, args=(tello,))

    # Start threads
    thread3.start()
    thread_recognize.start()

    thread3.join()
    thread_recognize.join()


def test_determine_place_on_video(tello):
    session_number = "demo_session_lab"
    image = tello.get_frame_read().frame
    engine_configs = EngineConfigurations({})
    engine_configs.add_configuration(
        AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_EXTRACTOR,
        "GoogLeNetPlaces365")
    engine = PlaceRecognitionEngine(PlaceRecognitionAlgorithm.PLACE_RECOGNITION_KNN, engine_configs)
    counter = 0
    tello.move_up(50)
    predictions = []

    current_rotation = 0
    while True:
        if current_rotation > 360:
            tello.land()
            break
        counter += 1
        image = tello.get_frame_read().frame
        result = engine.recognize_place(image)

        identified_place = "None"

        print(result)
        if result is not None and result.recognized_places is not None:
            print(result)
            predictions.append(result.recognized_places)
            path = r"C:\Users\Public\projects\drone-buddy-library\test\test_results\session_" + session_number
            file_name_path = (
                    path + "\\" + result.recognized_places + "_" + str(
                counter) + ".jpg")
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(file_name_path, image)
        current_rotation += 10
        tello.rotate_clockwise(10)

    cv2.imshow("Image" + identified_place + " _ ", image)
    cv2.waitKey(1)
    # Count occurrences of each prefix
    prefix_counts = Counter(predictions)
    print(prefix_counts)
    if prefix_counts is not None:
        print("Most common prefix:", prefix_counts.most_common(1)[0][0])
        if prefix_counts.most_common(1)[0][0] == "unknown":
            could_be = prefix_counts.most_common(2)
            print("this could be the : ", could_be)
            say_out_loud("There is a high chance this could be the " + str(could_be))


def keep_drone_in_air(drone_instance):
    moving_dir = -1
    while True:
        logger.log_info("Executing functions: ", "Drone is in the air")

        if drone_instance is not None and is_drone_in_air:
            print("battery: ", drone_instance.get_battery())
            print("temperature: ", drone_instance.get_temperature())
            if drone_instance.get_battery() < 20:
                drone_instance.land()
                logger.log_info("Executing functions: ", "Low battery, landing")

            if drone_instance.get_temperature() > 90:
                drone_instance.land()
                logger.log_info("Executing functions: ", "High temperature, landing")
            drone_instance.send_rc_control(0, 0, moving_dir, 0)
            moving_dir = moving_dir * -1

        # break
        time.sleep(4)  # Sleep to simulate work and prevent a tight loop


if __name__ == '__main__':
    test_face_rec_knn_recognize_place_by_video()

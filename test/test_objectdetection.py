import unittest
import time

import cv2

from dronebuddylib.atoms.intentrecognition.intent_recognition_engine import IntentRecognitionEngine
from dronebuddylib.atoms.objectdetection.mp_object_detection_impl import MPObjectDetectionImpl
import mediapipe as mp

from dronebuddylib.atoms.objectdetection.yolo_object_detection_impl import YOLOObjectDetectionImpl
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import Configurations, IntentRecognitionAlgorithm


# read input image


class TestObjectDetection(unittest.TestCase):

    def test_upload_image(self):
        image = cv2.imread('test_image.jpg')
        # image = cv2.imread('group.jpg')
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(Configurations.OBJECT_DETECTION_YOLO_VERSION, "yolov8n.pt")
        engine = YOLOObjectDetectionImpl(engine_configs)
        objects = engine.get_detected_objects(image)
        print("objects", objects.object_names)
        # to the students
        # new intent - what can be done?
        # higher level corespondence to the the natural language
        # describing a scene
        # what is the scene?
        # drone has a personality, memory,

        # assert len(labels) > 0 and 'chair' in labels


    def test_yolo(self):
        image = cv2.imread('test_image.jpg')
        # image = cv2.imread('group.jpg')
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(Configurations.OBJECT_DETECTION_YOLO_VERSION,
                                         "yolov8n.pt")
        engine = YOLOObjectDetectionImpl(engine_configs)
        objects = engine.get_detected_objects(image)
        print("objects", objects.object_names)
        # assert len(labels) > 0 and 'chair' in labels


    def test_mediapipe(self):
        # image = cv2.imread('test_image.jpg')
        mp_image = mp.Image.create_from_file('test_image.jpg')
        # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

        engine = MPObjectDetectionImpl(EngineConfigurations({}))
        objects = engine.get_detected_objects(mp_image)
        print("objects", objects.object_names)


    def test_simple_performance(self):
        # init yolo
        image = cv2.imread('test_image.jpg')
        mp_image = mp.Image.create_from_file('test_image.jpg')

        engine_configs_yolo = EngineConfigurations({})
        engine_configs_yolo.add_configuration(Configurations.OBJECT_DETECTION_YOLO_VERSION,
                                              "yolov8n.pt")

        yolo_engine = YOLOObjectDetectionImpl(engine_configs_yolo)
        mp_engine = MPObjectDetectionImpl(EngineConfigurations({}))

        yolo_start = time.time()
        yolo_labels = yolo_engine.get_detected_objects(image)
        yolo_end = time.time()

        mp_start = time.time()
        mp_labels = mp_engine.get_detected_objects(mp_image)
        mp_end = time.time()

        yolo_time = yolo_end - yolo_start
        mp_time = mp_end - mp_start

        # print("yolo V3 time", yolo_time)

        print(f"Yolo V3 time: {yolo_time:.2f} seconds")
        print(f"MP time: {mp_time:.2f} seconds")
        print(f"Yolo V3 labels: {', '.join(yolo_labels.object_names)}")
        print(f"MP labels: {', '.join(mp_labels.object_names)}")

        # check the time taken by yolo to return the labels


def test_me(self):
    print('test_me')


if __name__ == '__main__':
    unittest.main()

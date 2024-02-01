import unittest
import time

import cv2

from dronebuddylib import ObjectDetectionEngine
from dronebuddylib.atoms.objectdetection.mp_object_detection_impl import MPObjectDetectionImpl
import mediapipe as mp

from dronebuddylib.atoms.objectdetection.yolo_object_detection_impl import YOLOObjectDetectionImpl
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import AtomicEngineConfigurations, VisionAlgorithm


# read input image


class TestObjectDetection(unittest.TestCase):

    def test_basic_object_detection_yolo(self):
        image = cv2.imread(r'C:\Users\Public\projects\drone-buddy-library\test\test_images\test_image.jpg')
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(AtomicEngineConfigurations.OBJECT_DETECTION_YOLO_VERSION, "yolov8n.pt")
        object_engine = ObjectDetectionEngine(VisionAlgorithm.YOLO, engine_configs)
        # engine = YOLOObjectDetectionImpl(engine_configs)
        detected_objects = object_engine.get_detected_objects(image)
        print("objects", detected_objects.object_names)
        assert len(detected_objects.object_names) > 0  # Expecting at least one object to be detected
        assert 'potted plant' in detected_objects.object_names  # Expecting a person to be detected

    def test_object_detection_with_different_formats(self):
        for image_format in [r'C:\Users\Public\projects\drone-buddy-library\test\test_images\test_image.jpg',
                             r'C:\Users\Public\projects\drone-buddy-library\test\test_images\object_detection.png']:
            engine_configs = EngineConfigurations({})
            engine_configs.add_configuration(AtomicEngineConfigurations.OBJECT_DETECTION_YOLO_VERSION, "yolov8n.pt")
            image = cv2.imread(image_format)
            engine = YOLOObjectDetectionImpl(engine_configs)

            detected_objects = engine.get_detected_objects(image)

            assert len(detected_objects.object_names) > 0  # Expecting at least one object

    def test_object_detection_with_no_object_image(self):
        image = cv2.imread(r"C:\Users\Public\projects\drone-buddy-library\test\test_images\no_object.png")
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(AtomicEngineConfigurations.OBJECT_DETECTION_YOLO_VERSION, "yolov8n.pt")
        engine = YOLOObjectDetectionImpl(engine_configs)

        detected_objects = engine.get_detected_objects(image)
        print("objects", detected_objects.object_names)

        assert len(detected_objects.object_names) == 0  # Expecting no objects

    #
    def test_object_detection_with_low_resolution_image(self):
        image = cv2.imread(r'C:\Users\Public\projects\drone-buddy-library\test\test_images\blurry.png')
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(AtomicEngineConfigurations.OBJECT_DETECTION_YOLO_VERSION, "yolov8n.pt")
        engine = YOLOObjectDetectionImpl(engine_configs)

        detected_objects = engine.get_detected_objects(image)
        print("objects", detected_objects.object_names)
        assert len(detected_objects.object_names) != 0  # Expecting no objects

        # Assertions based on expected behavior with low-res images

    def test_basic_object_detection_mp(self):
        mp_image = mp.Image.create_from_file(
            r'C:\Users\Public\projects\drone-buddy-library\test\test_images\test_image.jpg')
        engine = MPObjectDetectionImpl(EngineConfigurations({}))
        detected_objects = engine.get_detected_objects(mp_image)
        print("objects", detected_objects.object_names)
        assert len(detected_objects.object_names) > 0  # Expecting at least one object to be detected
        assert 'potted plant' in detected_objects.object_names  # Expecting a person to be detected

    def test_object_detection_with_no_object_image_mp(self):

        mp_image = mp.Image.create_from_file(
            r"C:\Users\Public\projects\drone-buddy-library\test\test_images\no_object.png")
        engine = MPObjectDetectionImpl(EngineConfigurations({}))
        detected_objects = engine.get_detected_objects(mp_image)
        print("objects", detected_objects.object_names)
        assert len(detected_objects.object_names) == 0  # Expecting at least one object

    def test_object_detection_with_different_formats_mp(self):

        for image_format in [r'C:\Users\Public\projects\drone-buddy-library\test\test_images\test_image.jpg',
                             r'C:\Users\Public\projects\drone-buddy-library\test\test_images\object_detection.png']:
            mp_image = mp.Image.create_from_file(image_format)
            engine = MPObjectDetectionImpl(EngineConfigurations({}))
            detected_objects = engine.get_detected_objects(mp_image)
            print("objects", detected_objects.object_names)
            assert len(detected_objects.object_names) > 0  # Expecting at least one object

    def test_simple_performance(self):
        # init yolo
        image = cv2.imread('test_image.jpg')
        mp_image = mp.Image.create_from_file('test_image.jpg')

        engine_configs_yolo = EngineConfigurations({})
        engine_configs_yolo.add_configuration(AtomicEngineConfigurations.OBJECT_DETECTION_YOLO_VERSION,
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


if __name__ == '__main__':
    unittest.main()

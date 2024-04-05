import threading
import time
import unittest

import cv2
from djitellopy import Tello

from dronebuddylib.atoms.placerecognition.place_recognition_engine import PlaceRecognitionEngine
from dronebuddylib.models.engine_configurations import EngineConfigurations, logger
from dronebuddylib.models.enums import FaceRecognitionAlgorithm, AtomicEngineConfigurations, PlaceRecognitionAlgorithm


class TestFaceRecognition(unittest.TestCase):
    is_drone_in_air = False

    def test_face_rec_knn_add_place(self):

        tello = Tello()
        tello.connect()
        tello.streamon()
        tello.get_frame_read().frame
        tello.takeoff()
        self.is_drone_in_air = True
        print("battery: ", tello.get_battery())
        print("temperature: ", tello.get_temperature())
        thread1 = threading.Thread(target=self.add_place, args=("i4_level_5_kitchen", tello,))
        thread3 = threading.Thread(target=self.keep_drone_in_air, args=(tello,))

        # Start threads
        thread3.start()
        thread1.start()

        thread1.join()
        thread3.join()

    def keep_drone_in_air(self, drone_instance):
        moving_dir = -1
        while True:
            logger.log_info("Executing functions: ", "Drone is in the air")

            if drone_instance is not None and self.is_drone_in_air:
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

    def add_place(self, name, tello):
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_DRONE_INSTANCE,
                                         tello)
        engine_configs.add_configuration(AtomicEngineConfigurations.PLACE_RECOGNITION_KNN_USE_DRONE_TO_CREATE_DATASET,
                                         'True')
        engine = PlaceRecognitionEngine(PlaceRecognitionAlgorithm.PLACE_RECOGNITION_KNN, engine_configs)
        result = engine.remember_place(None, name, tello)
        print(result)
        tello.land()

    def test_face_rec_knn_create_memory(self):
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(AtomicEngineConfigurations.FACE_RECOGNITION_KNN_USE_DRONE_TO_CREATE_DATASET,
                                         "True")
        engine = PlaceRecognitionEngine(PlaceRecognitionAlgorithm.PLACE_RECOGNITION_KNN, engine_configs)
        result = engine.create_memory("Using densenet121")
        print(result)

    #
    def test_face_rec_knn_recognize_place(self):
        # image = cv2.imread( r'C:\Users\Public\projects\drone-buddy-library\test\test_images\test_pace.jpeg')
        image = cv2.imread( r'C:\Users\Public\projects\drone-buddy-library\test\test_images\random_bedroom_1.png')
        # image = cv2.imread( r'C:\Users\Public\projects\drone-buddy-library\test\test_images\random_bedroom.png')
        # image = cv2.imread( r'C:\Users\Public\projects\drone-buddy-library\test\test_images\random_forest.png')
        # image = cv2.imread( r'C:\Users\Public\projects\drone-buddy-library\test\test_images\i4_lab.jpeg')
        # image = cv2.imread( r'C:\Users\Public\projects\drone-buddy-library\test\test_images\i4_meeting_room.jpeg')
        # image = cv2.imread(r'C:\Users\Public\projects\drone-buddy-library\test\test_images\i4_kitchen.jpeg')
        # image = cv2.imread( r'C:\Users\augme\Downloads\archive\data\20095.jpg')

        engine_configs = EngineConfigurations({})

        engine = PlaceRecognitionEngine(PlaceRecognitionAlgorithm.PLACE_RECOGNITION_KNN, engine_configs)
        result = engine.recognize_place(image)
        # result = engine.recognize_place(r'C:\Users\Public\projects\drone-buddy-library\test\test_images\place_2.jpg')
        # result = engine.recognize_place(r'C:\Users\Public\projects\drone-buddy-library\test\test_images\test_pace.jpeg')
        print(result)
    #
    # def test_classifier(self):
    #     engine_configs = EngineConfigurations({})
    #     # engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECOGNITION_KNN, engine_configs)
    #     engine_configs = EngineConfigurations({})
    #
    #     engine = PlaceRecognitionEngine(PlaceRecognitionAlgorithm.PLACE_RECOGNITION_KNN, engine_configs)
    #     result = engine.test_classifier()
    #     print(result)


if __name__ == '__main__':
    unittest.main()

import unittest

import cv2

from dronebuddylib.atoms.facerecognition.face_recognition_engine import FaceRecognitionEngine
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import FaceRecognitionAlgorithm


class TestFaceRecognition(unittest.TestCase):

    def test_add_to_the_memory(self):
        file_path = r"C:\Users\Public\projects\drone-buddy-library\test\test_images\john.jpg"
        engine_configs = EngineConfigurations({})
        engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECC, engine_configs)
        result = engine.remember_face(file_path, "John")
        print(result)
        self.assertEqual(True, result)  # add assertion here

    def test_face_rec_with_image(self):
        image = cv2.imread(r'C:\Users\Public\projects\drone-buddy-library\test\test_images\jon_2.jpg')

        engine_configs = EngineConfigurations({})
        engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECC, engine_configs)
        result = engine.recognize_face(image)
        print(result)
        self.assertEqual(result[0], "John")  # add assertion here

    def test_face_rec_with_person_with_one_other(self):
        image = cv2.imread(r'C:\Users\Public\projects\drone-buddy-library\test\test_images\john_with_others.jpg')

        engine_configs = EngineConfigurations({})
        engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECC, engine_configs)
        result = engine.recognize_face(image)
        print(result)
        self.assertEqual(result[0], "John")  # add assertion here

    def test_face_rec_with_person_in_a_crowd(self):
        image = cv2.imread(r'C:\Users\Public\projects\drone-buddy-library\test\test_images\john_in_a_crowd.jpg')

        engine_configs = EngineConfigurations({})
        engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECC, engine_configs)
        result = engine.recognize_face(image)
        print(result)
        self.assertGreater(len(result), 0)  # add assertion here
        self.assertEqual(result[0], "John")  # add assertion here


if __name__ == '__main__':
    unittest.main()

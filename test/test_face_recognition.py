import unittest

import cv2

from dronebuddylib.atoms.facerecognition.face_recognition_engine import FaceRecognitionEngine
from dronebuddylib.atoms.facerecognition.face_recognition_knn_impl import FaceRecognitionKNNImpl
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import FaceRecognitionAlgorithm, AtomicEngineConfigurations


class TestFaceRecognition(unittest.TestCase):

    def test_add_to_the_memory(self):
        file_path = r"C:\Users\Public\projects\drone-buddy-library\test\test_images\john.jpg"
        engine_configs = EngineConfigurations({})
        engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECOGNITION_EUCLIDEAN, engine_configs)
        result = engine.remember_face(file_path, "John")
        print(result)
        self.assertEqual(True, result)  # add assertion here

    def test_face_rec_with_image(self):
        image = cv2.imread(r'C:\Users\Public\projects\drone-buddy-library\test\test_images\jon_2.jpg')

        engine_configs = EngineConfigurations({})
        engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECOGNITION_EUCLIDEAN, engine_configs)
        result = engine.recognize_face(image)
        print(result)
        self.assertEqual(result[0], "John")  # add assertion here

    def test_face_rec_with_person_with_one_other(self):
        image = cv2.imread(r'C:\Users\Public\projects\drone-buddy-library\test\test_images\john_with_others.jpg')

        engine_configs = EngineConfigurations({})
        engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECOGNITION_EUCLIDEAN, engine_configs)
        result = engine.recognize_face(image)
        print(result)
        self.assertEqual(result[0], "John")  # add assertion here

    def test_face_rec_with_person_in_a_crowd(self):
        image = cv2.imread(r'C:\Users\Public\projects\drone-buddy-library\test\test_images\john_in_a_crowd.jpg')

        engine_configs = EngineConfigurations({})
        engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECOGNITION_EUCLIDEAN, engine_configs)
        result = engine.recognize_face(image)
        print(result)
        self.assertGreater(len(result), 0)  # add assertion here
        self.assertEqual(result[0], "John")  # add assertion here

    def test_face_rec_knn_add_face(self):
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(AtomicEngineConfigurations.FACE_RECOGNITION_KNN_USE_DRONE_TO_CREATE_DATASET,
                                         "False")
        engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECOGNITION_KNN, engine_configs)
        result = engine.remember_face(None, "test")
        print(result)

    def test_face_rec_knn_create_memory(self):
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(AtomicEngineConfigurations.FACE_RECOGNITION_KNN_USE_DRONE_TO_CREATE_DATASET,
                                         "True")
        engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECOGNITION_KNN, engine_configs)
        result = engine.create_memory()
        print(result)

    def test_face_rec_knn_recognize_face(self):
        image = cv2.imread(
            r'C:\Users\Public\projects\drone-buddy-launcher\resources\images\img_1710768258.3202238.jpg')

        engine_configs = EngineConfigurations({})

        engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECOGNITION_KNN, engine_configs)
        result = engine.recognize_face(image)
        print(result)

    def test_classifier(self):
        engine_configs = EngineConfigurations({})
        # engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECOGNITION_KNN, engine_configs)
        engine = FaceRecognitionKNNImpl(engine_configs)
        result = engine.test_classifier()
        print(result)


if __name__ == '__main__':
    unittest.main()

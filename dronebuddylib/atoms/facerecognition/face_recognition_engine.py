from dronebuddylib.atoms.facerecognition.face_recognition_impl import FaceRecognitionImpl
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import FaceRecognitionAlgorithm


class FaceRecognitionEngine:
    def __init__(self, algorithm: FaceRecognitionAlgorithm, config: EngineConfigurations):
        self.face_recognition_model = algorithm

        if algorithm == FaceRecognitionAlgorithm.FACE_RECC:
            self.face_recognition_engine = FaceRecognitionImpl(config)

    def recognize_face(self, image):
        return self.face_recognition_engine.recognize_face(image)

    def remember_face(self, image, name):
        return self.face_recognition_engine.remember_face(image, name)

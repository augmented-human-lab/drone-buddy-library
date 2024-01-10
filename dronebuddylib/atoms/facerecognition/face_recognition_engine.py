from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import FaceRecognitionAlgorithm


class FaceRecognitionEngine:
    """
    The FaceRecognitionEngine class handles face recognition operations.
    """
    def __init__(self, algorithm: FaceRecognitionAlgorithm, config: EngineConfigurations):
        """
        Initialize the FaceRecognitionEngine class.

        Args:
            algorithm (FaceRecognitionAlgorithm): The algorithm to be used for face recognition.
            config (EngineConfigurations): The configurations for the engine.
        """
        self.face_recognition_model = algorithm

        if algorithm == FaceRecognitionAlgorithm.FACE_RECC:
            from dronebuddylib.atoms.facerecognition.face_recognition_impl import FaceRecognitionImpl
            self.face_recognition_engine = FaceRecognitionImpl(config)
        else:
            # Optionally handle other algorithms if you have any.
            raise ValueError("Unsupported face recognition algorithm")

    def recognize_face(self, image):
        """
        Recognize faces in an image.

        Args:
            image: The image containing faces to be recognized.

        Returns:
            A list of recognized faces.
        """
        return self.face_recognition_engine.recognize_face(image)

    def remember_face(self, image, name):
        """
        Remember a face by associating it with a name.

        Args:
            image: The image containing the face.
            name (str): The name to be associated with the face.

        Returns:
            True if the operation was successful, False otherwise.
        """
        return self.face_recognition_engine.remember_face(image, name)

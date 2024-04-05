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

        if algorithm == FaceRecognitionAlgorithm.FACE_RECOGNITION_EUCLIDEAN or algorithm == FaceRecognitionAlgorithm.FACE_RECOGNITION_EUCLIDEAN.name:
            from dronebuddylib.atoms.facerecognition.face_recognition_impl import FaceRecognitionImpl
            self.face_recognition_engine = FaceRecognitionImpl(config)
        elif algorithm == FaceRecognitionAlgorithm.FACE_RECOGNITION_KNN or algorithm == FaceRecognitionAlgorithm.FACE_RECOGNITION_KNN.name:
            from dronebuddylib.atoms.facerecognition.face_recognition_knn_impl import FaceRecognitionKNNImpl
            self.face_recognition_engine = FaceRecognitionKNNImpl(config)
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

    def create_memory(self):
        """
        Create a memory for the face recognition engine.

        Returns:
            A memory for the face recognition engine.
        """
        return self.face_recognition_engine.create_memory()

    def get_current_status(self):
        """
        Get the current status of the face recognition engine.

        Returns:
            The current status of the face recognition engine.
        """
        return self.face_recognition_engine.get_current_status()

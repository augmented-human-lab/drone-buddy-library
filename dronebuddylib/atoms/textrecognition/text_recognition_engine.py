from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import TextRecognitionAlgorithm
from dronebuddylib.utils.utils import logger


class TextRecognitionEngine:
    """
    The TextRecognitionEngine class handles text recognition operations.
    """

    def __init__(self, algorithm: TextRecognitionAlgorithm, config: EngineConfigurations):
        """
        Initialize the TextRecognitionEngine class.

        Args:
            algorithm (TextRecognitionAlgorithm): The algorithm to be used for text recognition.
            config (EngineConfigurations): The configurations for the engine.
        """
        self.text_recognition_model = algorithm

        if (algorithm == TextRecognitionAlgorithm.GOOGLE_VISION
                or algorithm == TextRecognitionAlgorithm.GOOGLE_VISION.name):
            logger.log_info(self.get_class_name(), 'Preparing to initialize Google Vision engine.')

            from dronebuddylib.atoms.textrecognition.google_text_recognition_impl import GoogleTextRecognitionImpl
            self.text_recognition_engine = GoogleTextRecognitionImpl(config)
        else:
            # Optionally handle other algorithms if you have any.
            raise ValueError("Unsupported face recognition algorithm")

    def get_class_name(self) -> str:
        """
        Returns the class name.

        Returns:
            str: The class name.
        """
        return 'TEXT_RECOGNITION_ENGINE'

    def recognize_text(self, image):
        """
        Recognize faces in an image.

        Args:
            image: The image containing faces to be recognized.

        Returns:
            A list of recognized faces.
        """
        return self.text_recognition_engine.recognize_text(image)

from google.cloud import vision

from dronebuddylib.atoms.textrecognition.i_text_recognition import ITextRecognition
from dronebuddylib.atoms.textrecognition.text_recognition_result import TextRecognitionResult
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.utils.logger import Logger

logger = Logger()


class GoogleTextRecognitionImpl(ITextRecognition):
    """
    Implementation of the IFaceRecognition interface using face_recognition library.
    """

    def get_required_params(self) -> list:
        return []

    def get_optional_params(self) -> list:
        return []

    def get_class_name(self) -> str:
        return 'TEXT_RECOGNITION_GOOGLE'

    def get_algorithm_name(self) -> str:
        return 'Google Text Recognition'

    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Initialize the FaceRecognitionImpl class.

        Args:
            engine_configurations (EngineConfigurations): The configurations for the engine.
        """
        super().__init__(engine_configurations)
        self.client = vision.ImageAnnotatorClient()
        logger.log_debug(self.get_class_name(), 'Initialized the Google Image Annotator Client.')

    def recognize_text(self, image_path) -> TextRecognitionResult:
        """
        Detects text within an image located at the specified path.

        This method reads the image file, creates an Image object, and uses the
        Google Cloud Vision API's text detection capabilities to find text within the image.

        Args:
            image_path (str): The file path of the image to be processed.

        Returns:
            An object containing the response from the Google Cloud Vision API. This object
            includes text annotations for detected text in the image.

        Raises:
            Google API client exceptions if the request fails.
        """
        logger.log_debug(self.get_class_name(), ' Detection started.')

        # Read the image file
        with open(image_path, 'rb') as image_file:
            content = image_file.read()

        # Create an image object
        image = vision.Image(content=content)
        logger.log_debug(self.get_class_name(), ' Detection Successful.')

        # Perform OCR
        response = self.client.text_detection(image=image)
        texts = response.text_annotations
        logger.log_info(self.get_class_name(), 'Found {} text in image'.format(len(texts)))
        logger.log_debug(self.get_class_name(), ' Detection completed.')

        if texts is None or len(texts) == 0:
            logger.log_info(self.get_class_name(), ' No text found in image')
            return TextRecognitionResult("", "", [])
        return TextRecognitionResult(response.full_text_annotation.text, texts[0].locale, texts)

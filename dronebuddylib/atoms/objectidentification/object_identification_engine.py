from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import VisionAlgorithm, ObjectRecognitionAlgorithm
from dronebuddylib.atoms.objectdetection.detected_object import ObjectDetectionResult
from dronebuddylib.utils.logger import Logger

logger = Logger()


class ObjectIdentificationEngine:

    def __init__(self, algorithm: ObjectRecognitionAlgorithm, config: EngineConfigurations):
        """
        Initializes the object detection engine with the specified algorithm and configuration.

        Args:
            algorithm (VisionAlgorithm): The vision algorithm to be used for object detection.
            config (EngineConfigurations): The configuration for the vision engine.
        """
        if algorithm == ObjectRecognitionAlgorithm.YOLO_TRANSFER_LEARNING or algorithm == ObjectRecognitionAlgorithm.YOLO_TRANSFER_LEARNING.name:
            logger.log_info(self.get_class_name(), 'Preparing to initialize YOLO object recognition.')
            from dronebuddylib.atoms.objectidentification.object_identification_siamese_impl import \
                ObjectRecognitionYOLOImpl
            self.vision_engine = ObjectRecognitionYOLOImpl(config)

    def get_class_name(self) -> str:
        """
        Returns the class name.

        Returns:
            str: The class name.
        """
        return 'OBJECT_IDENTIFICATION_ENGINE'

    def remember_object(self, image=None, type=None, name=None, drone_instance=None, on_start=None,
                        on_training_set_complete=None,
                        on_validation_set_complete=None):
        """
        Remember an object by associating it with a name, facilitating its future identification and recall.

        Args:
            image: The image containing the object.
            name (str): The name to be associated with the object.

        Returns:
            True if the operation was successful, False otherwise.
        """
        return self.vision_engine.remember_object(image, type, name)

    def identify_object(self, image):
        """
        Identify objects in an image, identifying and categorizing various objects depicted in the image.

        Args:
            image: The image containing objects to be recognized.

        Returns:
            A list of recognized objects, each potentially with associated metadata such as object name or coordinates.
        """
        return self.vision_engine.identify_object(image)

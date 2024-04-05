from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import PlaceRecognitionAlgorithm


class PlaceRecognitionEngine:
    """
    The PlaceRecognitionEngine class handles place recognition operations, facilitating the recognition and remembrance of places within images.
    """

    def __init__(self, algorithm: PlaceRecognitionAlgorithm, config: EngineConfigurations):
        """
        Initialize the PlaceRecognitionEngine class with a specific recognition algorithm and configuration settings.

        Args:
            algorithm (PlaceRecognitionAlgorithm): The algorithm to be used for place recognition.
            config (EngineConfigurations): The configurations for the recognition engine, specifying how the recognition process should be performed.
        """
        self.place_recognition_model = algorithm

        if (algorithm == PlaceRecognitionAlgorithm.PLACE_RECOGNITION_KNN or
                algorithm == PlaceRecognitionAlgorithm.PLACE_RECOGNITION_KNN.name):
            from dronebuddylib.atoms.placerecognition.place_recognition_knn_impl import PlaceRecognitionKNNImpl
            self.place_recognition_engine = PlaceRecognitionKNNImpl(config)
        else:
            # Optionally handle other algorithms if you have any.
            raise ValueError("Unsupported place recognition algorithm")

    def recognize_place(self, image):
        """
        Recognize places in an image, identifying and categorizing various locations depicted in the image.

        Args:
            image: The image containing places to be recognized.

        Returns:
            A list of recognized places, each potentially with associated metadata such as location name or coordinates.
        """
        return self.place_recognition_engine.recognize_place(image)

    def remember_place(self, image=None, name=None, drone_instance=None, on_start=None, on_training_set_complete=None,
                        on_validation_set_complete=None):
        """
        Remember a place by associating it with a name, facilitating its future identification and recall.

        Args:
            image: The image containing the place.
            name (str): The name to be associated with the place.

        Returns:
            True if the operation was successful, False otherwise.
        """
        return self.place_recognition_engine.remember_place(image, name)

    def create_memory(self ,changes=None):
        """
        Create a memory database or structure for the place recognition engine, optimizing future recognition tasks.

        Returns:
            A data structure or system representing the memory of the place recognition engine, useful for improving recognition efficiency and accuracy.
        """
        return self.place_recognition_engine.create_memory(changes)

    def get_current_status(self):
        """
        Get the current operational status of the place recognition engine, including any relevant metrics or state information.

        Returns:
            An object or description providing insights into the current status of the place recognition engine, which may include its operational state, any errors, or performance metrics.
        """
        return self.place_recognition_engine.get_current_status()

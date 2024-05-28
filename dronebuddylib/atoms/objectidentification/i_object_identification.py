from abc import abstractmethod

from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.i_dbl_function import IDBLFunction


class IObjectIdentification(IDBLFunction):
    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Constructor to initialize the object detection engine.

        Args:
            engine_configurations (EngineConfigurations): The configurations for the object detection engine.
        """
        self.engine_configurations = engine_configurations


    @abstractmethod
    def remember_object(self, object_name, image=None, image_folder_path=None, ) -> bool:
        """
        Remembers an object by associating it with a name.

        Args:
            image: The image containing the object.
            object_name (str): The name to be associated with the object.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        pass

    @abstractmethod
    def identify_object(self, image) -> str:
        """
        Identifies an object in the provided image.

        Args:
            image: The image in which to identify the object.

        Returns:
            str: The name of the identified object.
        """
        pass

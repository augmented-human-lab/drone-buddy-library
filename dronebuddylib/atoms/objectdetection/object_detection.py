from abc import ABC, abstractmethod

from dronebuddylib.models.engine_configurations import EngineConfigurations


class ObjectDetection(ABC):
    def __int__(self, engine_configurations: EngineConfigurations):
        self.engine_configurations = engine_configurations

    @abstractmethod
    def get_detected_objects(self, image) -> list:
        pass

    @abstractmethod
    def get_bounding_boxes_of_detected_objects(self, image) -> list:
        pass

    @abstractmethod
    def get_required_params(self) -> list:
        pass

    @abstractmethod
    def get_optional_params(self) -> list:
        pass

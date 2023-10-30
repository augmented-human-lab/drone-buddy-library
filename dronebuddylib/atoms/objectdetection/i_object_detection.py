from abc import ABC, abstractmethod

from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.i_dbl_function import IDBLFunction


class IObjectDetection(IDBLFunction):
    def __int__(self, engine_configurations: EngineConfigurations):
        self.engine_configurations = engine_configurations

    @abstractmethod
    def get_detected_objects(self, image) -> list:
        pass

    @abstractmethod
    def get_bounding_boxes_of_detected_objects(self, image) -> list:
        pass

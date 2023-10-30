from abc import abstractmethod

from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.i_dbl_function import IDBLFunction


class IFaceRecognition(IDBLFunction):
    def __init__(self, engine_configurations: EngineConfigurations):
        self.engine_configurations = engine_configurations

    @abstractmethod
    def recognize_face(self, image) -> list:
        pass

    @abstractmethod
    def remember_face(self, image, name) -> bool:
        pass

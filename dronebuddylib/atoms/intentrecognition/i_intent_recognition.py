from abc import ABC, abstractmethod

from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.i_dbl_function import IDBLFunction


class IIntentRecognition(IDBLFunction):
    def __int__(self, engine_configurations: EngineConfigurations):
        self.engine_configurations = engine_configurations

    @abstractmethod
    def get_resolved_intent(self, phrase: str) -> str:
        pass


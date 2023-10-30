from abc import ABC, abstractmethod

from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.i_dbl_function import IDBLFunction


class ISpeechGeneration(IDBLFunction):
    def __int__(self, engine_configurations: EngineConfigurations):
        self.engine_configurations = engine_configurations

    @abstractmethod
    def read_phrase(self, phrase) -> bool:
        pass

    @abstractmethod
    def change_voice(self, voice_id) -> bool:
        pass

    @abstractmethod
    def change_volume(self, volume) -> bool:
        pass

    @abstractmethod
    def change_rate(self, rate) -> bool:
        pass

    @abstractmethod
    def get_current_configs(self) -> dict:
        pass


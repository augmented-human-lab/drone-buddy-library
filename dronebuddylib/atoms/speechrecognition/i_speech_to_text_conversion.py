from abc import  abstractmethod

from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.i_dbl_function import IDBLFunction
from dronebuddylib.models.recognized_speech import RecognizedSpeechResult


class ISpeechToTextConversion(IDBLFunction):
    def __int__(self, engine_configurations: EngineConfigurations):
        self.engine_configurations = engine_configurations

    @abstractmethod
    def recognize_speech(self, audio) -> RecognizedSpeechResult:
        pass

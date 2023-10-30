from dronebuddylib.atoms.speechgeneration.tts_speech_generation_impl import TTSTextToSpeechEngineImpl
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.utils.enums import SpeechGenerationAlgorithm


class SpeechGenerationEngine:

    def __init__(self, algorithm: SpeechGenerationAlgorithm, config: EngineConfigurations):
        self.algorithm = algorithm
        if algorithm == SpeechGenerationAlgorithm.GOOGLE_TTS_OFFLINE:
            self.speech_generation_engine = TTSTextToSpeechEngineImpl(config)

    def read_phrase(self, phrase: str):
        return self.speech_generation_engine.read_phrase(phrase)

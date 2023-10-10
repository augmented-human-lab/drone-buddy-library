from dronebuddylib.atoms.speechgeneration.offline_speech_generation import OffLineTextToSpeechEngine
from dronebuddylib.atoms.speechgeneration.speech_configs import SpeechConfigs
from dronebuddylib.utils.enums import SpeechGenerationAlgorithm


class SpeechGenerationEngine:

    def __init__(self, algorithm: SpeechGenerationAlgorithm, speech_config: SpeechConfigs):
        self.algorithm = algorithm
        self.speech_config = speech_config
        if algorithm == SpeechGenerationAlgorithm.GOOGLE_TTS_OFFLINE:
            self.speech_generation_engine = OffLineTextToSpeechEngine(speech_config.rate, speech_config.volume,
                                                                      speech_config.voice_id)

    def read_aloud(self, phrase: str):
        return self.speech_generation_engine.generate_speech_and_play(phrase)

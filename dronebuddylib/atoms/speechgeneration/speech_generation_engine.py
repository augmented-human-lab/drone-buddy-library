from dronebuddylib.atoms.speechgeneration.offline_speech_generation import OffLineTextToSpeechEngine
from dronebuddylib.atoms.speechgeneration.speech_configs import SpeechConfigs
from dronebuddylib.utils.enums import SpeechGenerationAlgorithm


def read_aloud(algorithm: SpeechGenerationAlgorithm, speech_config: SpeechConfigs, frame):
    if algorithm == SpeechGenerationAlgorithm.GOOGLE_TTS_OFFLINE:
        speech_generation_engine = OffLineTextToSpeechEngine(speech_config.rate, speech_config.volume,
                                                             speech_config.voice_id)
        return speech_generation_engine.generate_speech_and_play(frame)

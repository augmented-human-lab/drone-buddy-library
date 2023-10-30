from dronebuddylib.atoms.speechrecognition.google_speech_2_text_conversion_impl import GoogleSpeechToTextConversionImpl
from dronebuddylib.atoms.speechrecognition.vosk_speech_2_text_conversion_impl import VoskSpeechToTextConversionImpl
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.utils.enums import SpeechRecognitionAlgorithm


class SpeechToTextEngine:
    def __init__(self, algorithm: SpeechRecognitionAlgorithm, speech_config: EngineConfigurations):
        self.algorithm = algorithm
        self.speech_config = speech_config
        if algorithm == SpeechRecognitionAlgorithm.GOOGLE_SPEECH_RECOGNITION:
            self.speech_conversion_engine = GoogleSpeechToTextConversionImpl(speech_config)

        if algorithm == SpeechRecognitionAlgorithm.VOSK_SPEECH_RECOGNITION:
            self.speech_conversion_engine = VoskSpeechToTextConversionImpl(speech_config)

    def recognize_speech(self, audio_steam):
        return self.speech_conversion_engine.recognize_speech(audio_steam)

from google.cloud import speech

from dronebuddylib.atoms.speechrecognition.i_speech_to_text_conversion import ISpeechToTextConversion
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import Configurations
from dronebuddylib.models.recognized_speech import RecognizedSpeechResult
from dronebuddylib.utils.utils import config_validity_check


class GoogleSpeechToTextConversionImpl(ISpeechToTextConversion):
    def get_class_name(self) -> str:
        return 'SPEECH_TO_TEXT_GOOGLE'

    def get_algorithm_name(self) -> str:
        return 'Google Speech to Text'

    def get_optional_params(self) -> list:
        return [Configurations.SPEECH_RECOGNITION_GOOGLE_SAMPLE_RATE_HERTZ,
                Configurations.SPEECH_RECOGNITION_GOOGLE_LANGUAGE_CODE,
                Configurations.SPEECH_RECOGNITION_GOOGLE_ENCODING]

    def get_required_params(self) -> list:
        return []

    def __init__(self, engine_configurations: EngineConfigurations):
        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())
        configs = engine_configurations.get_configurations_for_engine(self.get_class_name())

        self.sample_rate = configs.get(Configurations.SPEECH_RECOGNITION_GOOGLE_SAMPLE_RATE_HERTZ, 16000)
        self.language = configs.get(Configurations.SPEECH_RECOGNITION_GOOGLE_LANGUAGE_CODE, 'en-US')
        self.encoding = configs.get(Configurations.SPEECH_RECOGNITION_GOOGLE_ENCODING,
                                    speech.RecognitionConfig.AudioEncoding.LINEAR16)

        self.speech_conversion_engine = speech.SpeechClient()

    def recognize_speech(self, audio_steam) -> RecognizedSpeechResult:
        """
        Recognizes speechrecognition from an audio stream using the Google Cloud Speech-to-Text client.

        Args:
            audio_stream (bytes): The audio stream content to be recognized.

        Returns:
            speech.RecognizeResponse: The response containing recognized speechrecognition results.

        Example:
            audio_content = get_audio_stream_from_somewhere()
            response = recognize_speech(speech_client, audio_content)
            for result in response.results:
                print(f"Transcript: {result.alternatives[0].transcript}")
        """
        audio = self.speech_conversion_engine.RecognitionAudio(content=audio_steam)

        config = speech.RecognitionConfig(
            encoding=self.encoding,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language,
        )

        # Detects speechrecognition in the audio file
        response = self.speech_conversion_engine.recognize(config=config, audio=audio)

        for result in response.results:
            print(f"Transcript: {result.alternatives[0].transcript}")

        return RecognizedSpeechResult(response.results, response.total_billed_time)

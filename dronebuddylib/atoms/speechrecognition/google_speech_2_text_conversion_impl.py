from google.cloud import speech

from dronebuddylib.atoms.speechrecognition.i_speech_to_text_conversion import ISpeechToTextConversion
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import Configurations
from dronebuddylib.models.recognized_speech import RecognizedSpeechResult
from dronebuddylib.utils.utils import config_validity_check


class GoogleSpeechToTextConversionImpl(ISpeechToTextConversion):
    """
       This class is an implementation of the ISpeechToTextConversion interface for Google Cloud Speech-to-Text API.

       Attributes:
           sample_rate (int): The sample rate of the audio stream in hertz.
           language (str): The language code of the speech in the audio stream.
           encoding (speech.RecognitionConfig.AudioEncoding): The encoding type of the audio stream.
           speech_conversion_engine (speech.SpeechClient): The Google Cloud Speech-to-Text client.
       """

    def get_class_name(self) -> str:
        """
           Gets the class name.

           Returns:
               str: The class name.
           """
        return 'SPEECH_TO_TEXT_GOOGLE'

    def get_algorithm_name(self) -> str:
        """
            Gets the algorithm name.

            Returns:
                str: The algorithm name.
            """
        return 'Google Speech to Text'

    def get_optional_params(self) -> list:
        """
           Gets the list of optional parameters.

           Returns:
               list: The list of optional parameters.
           """
        return [Configurations.SPEECH_RECOGNITION_GOOGLE_SAMPLE_RATE_HERTZ,
                Configurations.SPEECH_RECOGNITION_GOOGLE_LANGUAGE_CODE,
                Configurations.SPEECH_RECOGNITION_GOOGLE_ENCODING]

    def get_required_params(self) -> list:
        """
             Gets the list of required parameters.

             Returns:
                 list: The list of required parameters.
             """
        return []

    def __init__(self, engine_configurations: EngineConfigurations):
        """
         Initializes the GoogleSpeechToTextConversionImpl class with the provided engine configurations.

         Args:
             engine_configurations (EngineConfigurations): The engine configurations containing necessary parameters.
         """
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
            Recognizes speech from an audio stream using the Google Cloud Speech-to-Text API.

            Args:
                audio_steam (bytes): The audio stream content to be recognized.

            Returns:
                RecognizedSpeechResult: The result containing recognized speech and total billed time.
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

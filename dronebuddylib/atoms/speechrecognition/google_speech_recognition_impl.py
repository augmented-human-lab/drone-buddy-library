import re
import sys

from google.cloud import speech

from dronebuddylib.atoms.speechrecognition.i_speech_recognition import ISpeechRecognition
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import AtomicEngineConfigurations
from dronebuddylib.atoms.speechrecognition.recognized_speech import RecognizedSpeechResult
from dronebuddylib.utils.utils import config_validity_check, logger


class GoogleSpeechRecognitionImpl(ISpeechRecognition):
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
        return [AtomicEngineConfigurations.SPEECH_RECOGNITION_GOOGLE_SAMPLE_RATE_HERTZ,
                AtomicEngineConfigurations.SPEECH_RECOGNITION_GOOGLE_LANGUAGE_CODE,
                AtomicEngineConfigurations.SPEECH_RECOGNITION_GOOGLE_ENCODING]

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
        super().__init__(engine_configurations)
        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())
        configs = engine_configurations.get_configurations_for_engine(self.get_class_name())

        self.sample_rate = configs.get(AtomicEngineConfigurations.SPEECH_RECOGNITION_GOOGLE_SAMPLE_RATE_HERTZ, 16000)
        self.language = configs.get(AtomicEngineConfigurations.SPEECH_RECOGNITION_GOOGLE_LANGUAGE_CODE, 'en-US')
        self.encoding = configs.get(AtomicEngineConfigurations.SPEECH_RECOGNITION_GOOGLE_ENCODING,
                                    speech.RecognitionConfig.AudioEncoding.LINEAR16)
        logger.log_info(self.get_class_name() + ':Initializing with model with ' + self.language + '')

        self.speech_conversion_engine = speech.SpeechClient()
        logger.log_debug(self.get_class_name() + ' :Initialized the Google Speech Recognition')

    def recognize_speech(self, audio_steam) -> RecognizedSpeechResult:
        """
            Recognizes speech from an audio stream using the Google Cloud Speech-to-Text API.

            Args:
                audio_steam (bytes): The audio stream content to be recognized.

            Returns:
                RecognizedSpeechResult: The result containing recognized speech and total billed time.
            """
        logger.log_debug(self.get_class_name() + ' :Recognition started.')

        config = speech.RecognitionConfig(
            encoding=self.encoding,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language,
            max_alternatives=1,
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config, interim_results=True
        )

        # Detects speechrecognition in the audio file
        response = self.speech_conversion_engine.streaming_recognize(streaming_config, audio_steam)
        logger.log_debug(self.get_class_name() + ' :Recognition Successful.')

        # return RecognizedSpeechResult(response, None)
        return response

    def listen_print_loop(responses: object) -> str:
        """Iterates through server responses and prints them.

        The responses passed is a generator that will block until a response
        is provided by the server.

        Each response may contain multiple results, and each result may contain
        multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
        print only the transcription for the top alternative of the top result.

        In this case, responses are provided for interim results as well. If the
        response is an interim one, print a line feed at the end of it, to allow
        the next result to overwrite it, until the response is a final one. For the
        final one, print a newline to preserve the finalized transcription.

        Args:
            responses: List of server responses

        Returns:
            The transcribed text.
        """
        num_chars_printed = 0
        for response in responses:
            if not response.results:
                continue

            # The `results` list is consecutive. For streaming, we only care about
            # the first result being considered, since once it's `is_final`, it
            # moves on to considering the next utterance.
            result = response.results[0]
            if not result.alternatives:
                continue

            # Display the transcription of the top alternative.
            transcript = result.alternatives[0].transcript

            # Display interim results, but with a carriage return at the end of the
            # line, so subsequent lines will overwrite them.
            #
            # If the previous result was longer than this one, we need to print
            # some extra spaces to overwrite the previous result
            overwrite_chars = " " * (num_chars_printed - len(transcript))

            if not result.is_final:
                sys.stdout.write(transcript + overwrite_chars + "\r")
                sys.stdout.flush()

                num_chars_printed = len(transcript)

            else:
                print(transcript + overwrite_chars)

                # Exit recognition if any of the transcribed phrases could be
                # one of our keywords.
                if re.search(r"\b(exit|quit)\b", transcript, re.I):
                    print("Exiting..")
                    break

                num_chars_printed = 0

            return transcript

import sys

from dronebuddylib.atoms.speechrecognition.i_speech_recognition import ISpeechRecognition
from dronebuddylib.models.enums import AtomicEngineConfigurations, LoggerColors
from dronebuddylib.atoms.speechrecognition.recognized_speech import RecognizedSpeechResult
import speech_recognition as speech_recognition

from dronebuddylib.utils.enums import SpeechRecognitionMultiAlgoAlgorithmSupportedAlgorithms
from dronebuddylib.utils.logger import Logger
from dronebuddylib.utils.utils import config_validity_check


class MultiAlgoSpeechToTextConversionImplementation(ISpeechRecognition):
    """
       Implementation of the ISpeechToTextConversion interface using multiple speech recognition algorithms.
       Refer to https://pypi.org/project/SpeechRecognition/ for more information.

       This class provides a way to utilize different speech-to-text conversion algorithms based on configuration settings. It supports Google Cloud Speech API, IBM Recognition, and Whisper. If the required libraries for these services are not installed, the corresponding functionality will not be available.

       Attributes:
           logger (Logger): An instance of the Logger for logging purposes.
           mic_timeout (int): The maximum number of seconds the microphone listens before timing out.
           ibm_key (str): The IBM API key for using IBM speech recognition.
           phrase_time_limit (int): The maximum duration for a single phrase before cutting off.
           algorithm (str): The name of the algorithm to use for speech recognition.
           speech_recognizer (speech_recognition.Recognizer): The speech recognizer instance.

       Methods:
           recognize_speech(audio): Converts speech from an audio source to text using the specified algorithm.
           get_required_params(): Returns a list of required parameter names for this implementation.
           get_optional_params(): Returns a list of optional parameter names for this implementation.
           get_class_name(): Returns a string representing the class name.
           get_algorithm_name(): Returns a string representing the name of the algorithm used.
       """

    """
        Google Cloud Speech Library for Python (for Google Cloud Speech API users):

            Google Cloud Speech library for Python is required if and only if you want to use the Google Cloud Speech API (recognizer_instance.recognize_google_cloud).

            If not installed, everything in the library will still work, except calling recognizer_instance.recognize_google_cloud will raise an RequestError.

            According to the official installation instructions, the recommended way to install this is using Pip: execute pip install google-cloud-speech (replace pip with pip3 if using Python 3).

        Whisper (for Whisper users):

            Whisper is required if and only if you want to use whisper (recognizer_instance.recognize_whisper).

            You can install it with python3 -m pip install git+https://github.com/openai/whisper.git soundfile.

        Whisper API (for Whisper API users):
            The library openai is required if and only if you want to use Whisper API (recognizer_instance.recognize_whisper_api).

            If not installed, everything in the library will still work, except calling recognizer_instance.recognize_whisper_api will raise an RequestError.

            You can install it with python3 -m pip install openai.

    """

    logger = Logger()

    def __init__(self, engine_configurations):
        """
               Initializes the MultiAlgoSpeechToTextConversionImplementation instance.

               Args:
                   engine_configurations (object): Configuration settings for the speech recognition engine.
               """
        super().__init__(engine_configurations)

        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())
        configs = engine_configurations.get_configurations_for_engine(self.get_class_name())

        self.mic_timeout = configs.get(AtomicEngineConfigurations.SPEECH_RECOGNITION_MULTI_ALGO_ALGO_MIC_TIMEOUT, 5)
        self.ibm_key = configs.get(AtomicEngineConfigurations.SPEECH_RECOGNITION_MULTI_ALGO_IBM_KEY, None)
        self.phrase_time_limit = configs.get(
            AtomicEngineConfigurations.SPEECH_RECOGNITION_MULTI_ALGO_ALGO_PHRASE_TIME_LIMIT, 10)
        self.algorithm = configs.get(AtomicEngineConfigurations.SPEECH_RECOGNITION_MULTI_ALGO_ALGORITHM_NAME,
                                     SpeechRecognitionMultiAlgoAlgorithmSupportedAlgorithms.GOOGLE.name)

        self.speech_recognizer = speech_recognition.Recognizer()

        self.logger.log_debug(self.get_class_name(), 'Speech Recognition Multi Algorithm Engine Initialized')

    def recognize_speech(self, audio) -> RecognizedSpeechResult:
        """
         Recognizes and converts speech from the given audio input to text using the configured speech recognition algorithm.

         Args:
             audio (AudioData): The audio data to be recognized.

         Returns:
             RecognizedSpeechResult: The result of the speech recognition process.
         """
        self.logger.log_debug(self.get_class_name(), ' :Recognition started.')

        audio = self.speech_recognizer.listen(audio, timeout=self.mic_timeout, phrase_time_limit=self.phrase_time_limit)

        self.logger.log_info(self.get_class_name(), "Passing audio for recognition")

        text = ""
        try:
            if self.algorithm == SpeechRecognitionMultiAlgoAlgorithmSupportedAlgorithms.GOOGLE.name:
                self.logger.log_info(self.get_class_name(), 'Using Google Speech Recognition')
                text = self.speech_recognizer.recognize_google(audio)
            elif self.algorithm == SpeechRecognitionMultiAlgoAlgorithmSupportedAlgorithms.IBM.name:
                self.logger.log_info(self.get_class_name(), 'Using IBM Recognition')
                text = self.speech_recognizer.recognize_ibm(audio, self.ibm_key)
            elif self.algorithm == SpeechRecognitionMultiAlgoAlgorithmSupportedAlgorithms.WHISPER.name:
                self.logger.log_info(self.get_class_name(), 'Using Whisper Recognition')
                text = self.speech_recognizer.recognize_whisper(audio)
            self.logger.log_debug(self.get_class_name(), ' :Recognition Successful.')

            return RecognizedSpeechResult(text, None)

        except speech_recognition.UnknownValueError:
            sys.stdout.write(LoggerColors.RED.value)
            self.logger.log_error(
                self.get_class_name(), self.algorithm + ' Speech Recognition could not understand audio')
            return RecognizedSpeechResult(None, None)
        except speech_recognition.RequestError as e:
            self.logger.log_error(self.get_class_name(),
                                  "Could not request results from " + self.algorithm
                                  + " Speech Recognition service; {0}".format(e))
            return RecognizedSpeechResult(None, None)
        except speech_recognition.WaitTimeoutError:
            self.logger.log_error(self.get_class_name(), "Timeout error, rerunning...")
            self.recognize_speech(audio)

    def get_required_params(self) -> list:
        """
                Gets a list of required configuration parameters for this speech-to-text conversion implementation.

                Returns:
                    list: A list of required parameter names.
                """
        return [AtomicEngineConfigurations.SPEECH_RECOGNITION_MULTI_ALGO_ALGORITHM_NAME]

    def get_optional_params(self) -> list:
        """
              Gets a list of optional configuration parameters for this speech-to-text conversion implementation.

              Returns:
                  list: A list of optional parameter names.
              """
        return [AtomicEngineConfigurations.SPEECH_RECOGNITION_MULTI_ALGO_ALGO_MIC_TIMEOUT,
                AtomicEngineConfigurations.SPEECH_RECOGNITION_MULTI_ALGO_ALGO_PHRASE_TIME_LIMIT,
                AtomicEngineConfigurations.SPEECH_RECOGNITION_MULTI_ALGO_IBM_KEY,
                ]

    def get_class_name(self) -> str:
        """
               Gets the class name of this speech-to-text conversion implementation.

               Returns:
                   str: The class name.
               """
        return 'SPEECH_RECOGNITION_MULTI_ALGO'

    def get_algorithm_name(self) -> str:
        """
               Gets the name of the algorithm used for speech-to-text conversion in this implementation.

               Returns:
                   str: The algorithm name.
               """
        return 'Multi Algorithm Text to Speech'

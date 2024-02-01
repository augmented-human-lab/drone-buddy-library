import logging
import pkg_resources

from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import IntentRecognitionAlgorithm, DroneCommands
from dronebuddylib.atoms.intentrecognition.recognized_intent_result import RecognizedIntent
from dronebuddylib.utils import FileWritingException
from dronebuddylib.utils.utils import logger


class IntentRecognitionEngine:
    ACTION_FILE_PATH = pkg_resources.resource_filename(__name__, "/resources/intents.txt")
    """
    A high-level engine for intent recognition leveraging various algorithms.

    Attributes:
        intent_recognizer (IIntentRecognition): An instance of the intent recognition algorithm chosen.
    """

    def __init__(self, algorithm: IntentRecognitionAlgorithm, config: EngineConfigurations):
        """
        Initialize the IntentRecognitionEngine with a given algorithm and configuration.

        Args:
            algorithm (IntentRecognitionAlgorithm): The algorithm to be used for intent recognition.
            config (EngineConfigurations): Configuration parameters required for the chosen algorithm.
        """
        if self.get_current_intents().get(DroneCommands.TAKE_OFF.name) is None:
            try:
                with open(self.ACTION_FILE_PATH, 'a') as file:
                    list_actions = [e for e in DroneCommands]
                    for action in list_actions:
                        file.write(action.name + "=" + action.value + '\n')
            except IOError as e:
                logging.error(self.get_class_name(),"Error while writing default actions to the file: %s", e)
                raise FileWritingException("Error while writing default actions to the file.")

        if algorithm == IntentRecognitionAlgorithm.CHAT_GPT or algorithm == IntentRecognitionAlgorithm.CHAT_GPT.name:
            logger.log_info(self.get_class_name(), 'Preparing to initialize Chat GPT intent recognition engine.')
            #  import only if needed
            from dronebuddylib.atoms.intentrecognition.gpt_intent_recognition_impl import GPTIntentRecognitionImpl
            self.intent_recognizer = GPTIntentRecognitionImpl(config)
        elif algorithm == IntentRecognitionAlgorithm.SNIPS_NLU or algorithm == IntentRecognitionAlgorithm.SNIPS_NLU.name:
            #  import only if needed
            logger.log_info(self.get_class_name(), 'Preparing to initialize SNIPS NLU intent recognition engine.')

            from dronebuddylib.atoms.intentrecognition.snips_intent_recognition_impl import SNIPSIntentRecognitionImpl

            self.intent_recognizer = SNIPSIntentRecognitionImpl(config)
        else:
            raise ValueError("Invalid intent recognition algorithm specified.")

    def get_class_name(self) -> str:
        """
        Returns the class name.

        Returns:
            str: The class name.
        """
        return 'INTENT_RECOGNITION_ENGINE'

    def recognize_intent(self, text: str) -> RecognizedIntent:
        """
        Recognize the intent from the provided text using the configured algorithm.

        Args:
            text (str): The input text from which intent needs to be recognized.

        Returns:
            RecognizedIntent: Recognized intent.
        """

        return self.intent_recognizer.get_resolved_intent(text)

    def get_current_intents(self) -> dict:
        """
        Retrieve the current intents and their descriptions from the intent file.

        Returns:
            dict: A dictionary containing intents as keys and their descriptions as values.
        """
        try:
            with open(self.ACTION_FILE_PATH, "r") as file:
                lines = file.readlines()
                lines_without_newline = [line.rstrip('\n') for line in lines]
                intent_list = [line for line in lines_without_newline if line]
                intent_dict = {}
                for intent in intent_list:
                    intent_name, intent_description = intent.split("=")
                    intent_dict[intent_name] = intent_description
                return intent_dict
        except FileNotFoundError as e:
            logging.error(self.get_class_name(), "The specified file is not found: %s", e)
            raise FileNotFoundError("The specified file is not found.") from e

    def introduce_new_intent(self, intent: str, description: str) -> bool:
        """
        Add a new intent and its description to the intent file.

        Args:
            intent (str): The new intent to be added.
            description (str): The description of the new intent.

        Returns:
            bool: True if the new intent was successfully added, False otherwise.
        """
        try:
            text_file_path = pkg_resources.resource_filename(__name__, "resources/intentrecognition/intents.txt")
            with open(text_file_path, 'a') as file:
                file.write(intent + "=" + description + '\n')
            return True
        except IOError as e:
            logging.error(self.get_class_name(), "Error while writing to the file: %s", e)
            raise FileWritingException("Error while writing to the file: " + intent) from e

import logging

import pkg_resources

from dronebuddylib.atoms.intentrecognition.gpt_intent_recogntion import GPTIntentRecognition
from dronebuddylib.atoms.intentrecognition.snips_intent_recognition_impl import SNIPSIntentRecognitionImpl

from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import IntentRecognitionAlgorithm, DroneCommands
from dronebuddylib.models.intent import Intent
from dronebuddylib.utils import FileWritingException


class IntentRecognitionEngine:
    ACTION_FILE_PATH = pkg_resources.resource_filename(__name__, "/resources/intents.txt")

    """
    A high-level engine for intent recognition leveraging various algorithms.

    Attributes:
        intent_recognizer: An instance of the intent recognition algorithm chosen.

    Args:
        algorithm (IntentRecognitionAlgorithm): The algorithm to be used for intent recognition.
        config (IntentConfigs): Configuration parameters required for the chosen algorithm.

    Examples:
        >>> engine = IntentRecognitionEngine(IntentRecognitionAlgorithm.CHAT_GPT, config)
        >>> intent = engine.recognize_intent("Turn off the lights.")
    """

    def __init__(self, algorithm: IntentRecognitionAlgorithm, config: EngineConfigurations):
        """
        Initialize the IntentRecognitionEngine with a given algorithm and configuration.

        Args:
            algorithm (IntentRecognitionAlgorithm): The algorithm to be used for intent recognition.
            config (dict): Configuration parameters required for the chosen algorithm.
        """
        if self.get_current_intents().get(DroneCommands.TAKE_OFF.name) is None:
            # add intent to the intent list
            try:
                with open(self.ACTION_FILE_PATH, 'a') as file:
                    list_actions = [e for e in DroneCommands]
                    for action in list_actions:
                        file.write(action.name + "=" + action.value + '\n')
            except IOError:
                logging.error("Error while writing default actions to the file : ")
                raise FileWritingException("Error while writing default actions to the file : ")

        if algorithm == IntentRecognitionAlgorithm.CHAT_GPT:
            self.intent_recognizer = GPTIntentRecognition(config)
        if algorithm == IntentRecognitionAlgorithm.SNIPS_NLU:
            self.intent_recognizer = SNIPSIntentRecognitionImpl(config)

    def recognize_intent(self, text: str) -> Intent:
        """
        Recognize the intent from the provided text using the configured algorithm.

        Args:
            text (str): The input text from which intent needs to be recognized.

        Returns:
            Intent: Recognized intent.

        Examples:
            >>> engine = IntentRecognitionEngine(IntentRecognitionAlgorithm.CHAT_GPT, config)
            >>> intent = engine.recognize_intent("What's the weather today?")
            "get_weather"
        """
        return self.intent_recognizer.get_resolved_intent(text)

    def get_current_intents(self) -> dict:

        try:
            with open(self.ACTION_FILE_PATH, "r") as file:
                # Read the contents of the file line by line
                lines = file.readlines()
                lines_without_newline = [line.rstrip('\n') for line in lines]
                intent_list = [line for line in lines_without_newline if line]
                intent_dict = {}
                for intent in intent_list:
                    intent_name, intent_description = intent.split("=")
                    intent_dict[intent_name] = intent_description
                return intent_dict
        except FileNotFoundError as e:
            raise FileNotFoundError("The specified file is not found.", e) from e
        # Return the list of fields

    def add_new_intent(self, intent: str, description: str) -> bool:

        # add intent to the intent list
        try:
            text_file_path = pkg_resources.resource_filename(__name__, "resources/intentrecognition/intents.txt")
            with open(text_file_path, 'a') as file:
                file.write(intent + "=" + description + '\n')
        except IOError:
            logging.error("Error while writing to the file : ", intent)
            raise FileWritingException("Error while writing to the file : " + intent)
        return True

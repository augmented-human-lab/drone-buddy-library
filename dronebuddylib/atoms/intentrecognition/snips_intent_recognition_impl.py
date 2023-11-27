import json
from abc import ABC
import pkg_resources
from snips_nlu import SnipsNLUEngine

from dronebuddylib.atoms.intentrecognition.i_intent_recognition import IIntentRecognition
from dronebuddylib.exceptions.intent_resolution_exception import IntentResolutionException
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import Configurations
from dronebuddylib.models.intent import Intent
from dronebuddylib.utils import FileWritingException
from dronebuddylib.utils.utils import config_validity_check
import logging


class SNIPSIntentRecognitionImpl(IIntentRecognition):
    """
    Implementation of intent recognition using the Snips NLU engine.

    Attributes:
        engine (SnipsNLUEngine): The Snips NLU engine used for intent recognition.

    Methods:
        get_class_name: Returns the class name.
        get_algorithm_name: Returns the name of the algorithm.
        get_resolved_intent: Parses text to detect intent and associated slots.
        add_new_intent: Adds a new intent to the intent file.
        get_required_params: Returns a list of required configuration parameters.
        get_optional_params: Returns a list of optional configuration parameters.
    """

    def get_class_name(self) -> str:
        """Returns the class name."""
        return 'INTENT_RECOGNITION_SNIPS'

    def get_algorithm_name(self) -> str:
        """Returns the name of the algorithm."""
        return 'SNIPS Intent Recognition'

    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Initialize the SNIPSIntentRecognitionImpl with the given configurations.

        Args:
            engine_configurations (EngineConfigurations): The configurations for the engine.
        """
        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())

        # Use the default dataset path if not provided
        if engine_configurations.get_configurations_for_engine(self.get_class_name()).get(
                Configurations.INTENT_RECOGNITION_SNIPS_NLU_DATASET_PATH.name) is None:
            dataset_path = pkg_resources.resource_filename(__name__, "resources/intentrecognition/dataset.json")
            logging.debug(self.get_class_name() + ': initialized with default dataset')
        else:
            dataset_path = engine_configurations.get_configurations_for_engine(self.get_class_name()).get(
                Configurations.INTENT_RECOGNITION_SNIPS_NLU_DATASET_PATH.name)

        engine = SnipsNLUEngine(config=engine_configurations.get_configurations_for_engine(self.get_class_name()).get(
            Configurations.INTENT_RECOGNITION_SNIPS_LANGUAGE_CONFIG.name))

        with open(dataset_path) as fh:
            dataset = json.load(fh)
            logging.debug(self.get_class_name() + ':Training set loaded from the json file')

        engine.fit(dataset)
        logging.debug(self.get_class_name() + ': Model fitted to the training set')

        self.engine = engine
        logging.debug(self.get_class_name() + ': Initialized intent recognition engine')

    def get_resolved_intent(self, phrase: str) -> Intent:
        """
        Parses text to detect intent and associated slots.

        Args:
            phrase (str): The text to be parsed.

        Returns:
            Intent: The detected intent and associated slots.
        """
        intent = self.engine.parse(phrase)
        try:
            return Intent(intent['intent']['intentName'], intent['slots'], intent['intent']['probability'])
        except KeyError:
            raise IntentResolutionException("Intent could not be resolved.")

    def add_new_intent(self, intent: str, description: str) -> bool:
        """
        Adds a new intent and its description to the intent file.

        Args:
            intent (str): The new intent.
            description (str): The description of the new intent.

        Returns:
            bool: True if the intent was successfully added, False otherwise.
        """
        try:
            text_file_path = pkg_resources.resource_filename(__name__, "resources/intentrecognition/intents.txt")
            with open(text_file_path, 'a') as file:
                file.write(intent + "=" + description + '\n')
            return True
        except IOError:
            logging.error("Error while writing to the file: %s", intent)
            raise FileWritingException("Error while writing to the file: " + intent)

    def get_required_params(self) -> list:
        """Returns a list of required configuration parameters."""
        return [Configurations.INTENT_RECOGNITION_SNIPS_NLU_DATASET_PATH,
                Configurations.INTENT_RECOGNITION_SNIPS_LANGUAGE_CONFIG]

    def get_optional_params(self) -> list:
        """Returns a list of optional configuration parameters."""
        pass

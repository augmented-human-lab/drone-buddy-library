import json
import pkg_resources
from snips_nlu import SnipsNLUEngine

from dronebuddylib.atoms.intentrecognition.i_intent_recognition import IIntentRecognition
from dronebuddylib.exceptions.intent_resolution_exception import IntentResolutionException
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import AtomicEngineConfigurations
from dronebuddylib.atoms.intentrecognition.recognized_intent import RecognizedIntent, RecognizedEntities
from dronebuddylib.utils import FileWritingException
from dronebuddylib.utils.utils import config_validity_check, logger


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

    def introduce_new_intents(self, new_intents: dict) -> bool:
        """
        Introduces new intents to the intent recognition system.

        Args:
            new_intents (dict): The new intents to be added.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        try:
            text_file_path = pkg_resources.resource_filename(__name__, "resources/intentrecognition/intents.txt")
            with open(text_file_path, 'a') as file:
                for intent in new_intents:
                    file.write(intent + "=" + new_intents[intent] + '\n')
            return True
        except IOError:
            logger.log_error(self.get_class_name(), "Error while writing to the file: %s", intent)
            raise FileWritingException("Error while writing to the file: " + intent)

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
        super().__init__(engine_configurations)
        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())

        # Use the default dataset path if not provided
        if engine_configurations.get_configurations_for_engine(self.get_class_name()).get(
                AtomicEngineConfigurations.INTENT_RECOGNITION_SNIPS_NLU_DATASET_PATH.name) is None:
            dataset_path = pkg_resources.resource_filename(__name__, "resources/intentrecognition/dataset.json")
            logger.log_debug(self.get_class_name(), 'initialized with default dataset')
        else:
            dataset_path = engine_configurations.get_configurations_for_engine(self.get_class_name()).get(
                AtomicEngineConfigurations.INTENT_RECOGNITION_SNIPS_NLU_DATASET_PATH.name)

        engine = SnipsNLUEngine(config=engine_configurations.get_configurations_for_engine(self.get_class_name()).get(
            AtomicEngineConfigurations.INTENT_RECOGNITION_SNIPS_LANGUAGE_CONFIG.name))

        with open(dataset_path) as fh:
            dataset = json.load(fh)
            logger.log_debug(self.get_class_name(), 'Training set loaded from the json file')

        engine.fit(dataset)
        logger.log_debug(self.get_class_name(), ' Model fitted to the training set')

        self.engine = engine
        logger.log_debug(self.get_class_name(), ' Initialized intent recognition engine')

    def get_resolved_intent(self, phrase: str) -> RecognizedIntent:
        """
        Parses text to detect intent and associated slots.

        Args:
            phrase (str): The text to be parsed.

        Returns:
            RecognizedIntent: The detected intent and associated slots.
        """
        logger.log_debug(self.get_class_name(), ' Detection started.')

        intent = self.engine.parse(phrase)
        logger.log_debug(self.get_class_name(), ' Detection Successful.')

        try:
            formatted_intent = RecognizedIntent(intent['intent']['intentName'], [], intent['intent']['probability'],
                                                False)
            entity_list = []
            for slot in intent['slots']:
                entity_list.append(RecognizedEntities(slot['entity'], slot['rawValue']))
                if slot['entity'] == 'DroneName':
                    formatted_intent.addressed_to = True

            formatted_intent.set_entities(entity_list)
            logger.log_debug(self.get_class_name(), ' Detection completed.')
            return formatted_intent
        except KeyError:
            raise IntentResolutionException("Intent could not be resolved.", 500)

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
            text_file_path = pkg_resources.resource_filename(__name__, "intentrecognition/resources/intents.txt")
            with open(text_file_path, 'a') as file:
                file.write(intent + "=" + description + '\n')
            return True
        except IOError:
            logger.log_error(self.get_class_name(), " Error while writing to the file: %s", intent)
            raise FileWritingException("Error while writing to the file: " + intent)

    def get_required_params(self) -> list:
        """Returns a list of required configuration parameters."""
        return []

    def get_optional_params(self) -> list:
        """Returns a list of optional configuration parameters."""
        return [AtomicEngineConfigurations.INTENT_RECOGNITION_SNIPS_NLU_DATASET_PATH,
                AtomicEngineConfigurations.INTENT_RECOGNITION_SNIPS_LANGUAGE_CONFIG]

import json
from abc import ABC

import pkg_resources
from snips_nlu import SnipsNLUEngine

from dronebuddylib.atoms.intentrecognition.i_intent_recognition import IIntentRecognition
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import Configurations
from dronebuddylib.models.intent import Intent
from dronebuddylib.utils import FileWritingException
from dronebuddylib.utils.utils import config_validity_check
import logging


class SNIPSIntentRecognitionImpl(IIntentRecognition):
    def get_class_name(self) -> str:
        return 'INTENT_RECOGNITION_SNIPS'

    def get_algorithm_name(self) -> str:
        return 'SNIPS Intent Recognition'

    def __init__(self, engine_configurations: EngineConfigurations):
        # Load the SnipsNLUEngine with the provided configuration

        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())

        # If dataset path is not provided, use the default path
        if engine_configurations.get_configurations_for_engine(self.get_class_name()).get(
                Configurations.INTENT_RECOGNITION_SNIPS_NLU_DATASET_PATH.name) is None:
            dataset_path = pkg_resources.resource_filename(__name__, "resources/intentrecognition/dataset.json")
            logging.debug(self.get_class_name() + ': initialized with default dataset')
        else:
            dataset_path = engine_configurations.get_configurations_for_engine(self.get_class_name()).get(
                Configurations.INTENT_RECOGNITION_SNIPS_NLU_DATASET_PATH.name)

        # Load the SnipsNLUEngine with the provided configuration
        engine = SnipsNLUEngine(config=engine_configurations.get_configurations_for_engine(self.get_class_name()).get(
            Configurations.INTENT_RECOGNITION_SNIPS_LANGUAGE_CONFIG.name))

        # Load the dataset file in JSON format

        with open(
                dataset_path) as fh:
            dataset = json.load(fh)
            logging.debug(self.get_class_name() + ':Training set loaded from the json file')

        # Fit the dataset to the engine
        engine.fit(dataset)
        logging.debug(self.get_class_name() + ': Model fitted to the training set')

        self.engine = engine
        logging.debug(self.get_class_name() + ': Initialized intent recognition engine')

    def get_resolved_intent(self, phrase: str) -> Intent:
        """
           Given a trained SnipsNLUEngine and a string of text, this function parses the text
           and returns a dictionary representing the detected intent and associated slots.

           Args:
               phrase: A string of text to be parsed by the NLU engine

           Returns:
               A dictionary representing the detected intent and associated slots, with the following keys:
                   - intent: An object containing intentName, probability.
                   - slots: A dictionary containing key-value pairs of detected slots
           """
        intent = self.engine.parse(phrase)
        return Intent(intent.intent, intent.slots, intent.intent.probability)

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

    def get_required_params(self) -> list:
        return [Configurations.INTENT_RECOGNITION_SNIPS_NLU_DATASET_PATH,
                Configurations.INTENT_RECOGNITION_SNIPS_LANGUAGE_CONFIG
                ]

    def get_optional_params(self) -> list:
        pass

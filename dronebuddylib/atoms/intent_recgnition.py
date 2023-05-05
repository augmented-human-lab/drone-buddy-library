import json

import pkg_resources
from snips_nlu import SnipsNLUEngine
from snips_nlu.default_configs import CONFIG_EN

from utils.logging_config import get_logger

# Get an instance of a logger
logger = get_logger()

'''
in order to get the trained json file follow the https://snips-nlu.readthedocs.io/en/latest/tutorial.html
'''


def init_intent_recognition_engine(dataset_path: str = None, config: str = CONFIG_EN):
    """
       Initialize the intent recognition engine using the provided dataset file.

       Args:
           dataset_path (str): Path to the JSON dataset file containing the intents and their corresponding utterances.
           config (str): The configuration to use for the SnipsNLUEngine.default is english

       Returns:
           SnipsNLUEngine: A trained instance of the SnipsNLUEngine used for intent recognition.
       """

    # If dataset path is not provided, use the default path
    if dataset_path is None:
        dataset_path = pkg_resources.resource_filename(__name__, "resources/intentrecognition/dataset.json")
        logger.info('Intent Recognition : Loading default dataset')

    # Load the SnipsNLUEngine with the provided configuration
    engine = SnipsNLUEngine(config=config)

    # Load the dataset file in JSON format
    # with open(
    #         dataset_path,
    #         encoding='utf-16') as fh:

    with open(
            dataset_path) as fh:
        dataset = json.load(fh)
        logger.debug('Intent Recognition : Training set loaded from the json file')

    # Fit the dataset to the engine
    engine.fit(dataset)
    logger.debug('Intent Recognition : Model fitted to the training set')

    logger.info('Intent Recognition : Initialized intent recognition engine')
    # Return the trained intent recognition engine
    return engine


def recognize_intent(engine: SnipsNLUEngine, text: str):
    """
       Given a trained SnipsNLUEngine and a string of text, this function parses the text
       and returns a dictionary representing the detected intent and associated slots.

       Args:
           engine: An instance of SnipsNLUEngine, which has been trained on a dataset of intents
           text: A string of text to be parsed by the NLU engine

       Returns:
           A dictionary representing the detected intent and associated slots, with the following keys:
               - input: the given input text
               - intent: An object containing intentName, probability.
               - slots: A dictionary containing key-value pairs of detected slots
       """
    return engine.parse(text)


if __name__ == '__main__':
    engine = init_intent_recognition_engine()
    recognized_intent = recognize_intent(engine, "can you please go up")
    print(recognized_intent)

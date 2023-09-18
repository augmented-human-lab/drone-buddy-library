import json

import pkg_resources
from snips_nlu import SnipsNLUEngine
from snips_nlu.default_configs import CONFIG_EN

from dronebuddylib.utils.logging_config import get_logger

# Get an instance of a logger
logger = get_logger()

'''
in order to get the trained json file follow the https://snips-nlu.readthedocs.io/en/latest/tutorial.html
'''


def init_intent_recognition_engine_1(dataset_path: str = None, config: str = CONFIG_EN):
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


def get_intent_name(intent, threshold=0.5):
    """
      Retrieves the name of the recognized intent based on the provided intent object.

      Args:
          intent (dict): The intent object containing information about the recognized intent.
          threshold (float): The probability threshold for considering the intent as valid.

      Returns:
          str or None: The name of the recognized intent if its probability is above the threshold, otherwise None.

      Example:
          intent_name = get_intent_name(intent_obj, threshold=0.6)
      """
    if intent is None:
        return None
    if intent['intent']['probability'] < threshold:
        return None
    return intent['intent']['intentName']


def get_mentioned_entities(intent):
    """
     Retrieves the key-value pairs from the slots of an intent.

     Args:
         intent (dict): The intent object containing slots.

     Returns:
         dict: A dictionary containing the key-value pairs extracted from the slots.
               Returns None if the intent is None, slots are None, or if there are no slots.

     """
    if intent is None:
        return None
    if intent['slots'] is None:
        return None
    if len(intent['slots']) == 0:
        return None
    # Get the slots from the intent object
    slots = intent.get("slots")
    # Initialize an empty dictionary for storing the key-value pairs
    slot_values = {}
    # Extract the key-value pairs from the slots and store them in the slot_values dictionary
    for slot in slots:
        slot_values[slot.get("entity")] = slot.get("rawValue")

    return slot_values


def is_addressed_to_drone(intent, name='sammy', similar_pronunciation=None):
    """
    Checks if the intent is addressed to the drone

    Args:
        intent (dict): The intent object containing slots.
        name (str): The name in which the drone is to be addressed, this should be the same name that the intent
         classifier ia trained with.
        similar_pronunciation (list): A list of names that sound similar to the name of the drone

    Returns:
        bool: True if the intent is addressed to the drone, False otherwise
    """

    slot_values = get_mentioned_entities(intent)
    if slot_values is None and name in intent.get("input").casefold():
        return True
    if slot_values is None:
        return False
    if 'address' in slot_values.keys() and (
            slot_values['address'].casefold() == name or name in intent.get("input").casefold()):
        return True
    if similar_pronunciation is not None and 'address' in slot_values.keys() and (
            slot_values['address'].casefold() in similar_pronunciation):
        return True
    else:
        return False

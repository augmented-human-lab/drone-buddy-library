import json

import pkg_resources
from snips_nlu import SnipsNLUEngine
from snips_nlu.default_configs import CONFIG_EN

from dronebuddylib.utils.logging_config import get_logger

logger = get_logger()


class OfflineIntentRecognitionEngine:
    """
    Offline intent recognition system leveraging Snips NLU for processing user intents.

    Provides functionalities to recognize intents, retrieve intent names, mentioned entities,
    and determine if a given intent is addressed to the drone.

    Attributes:
        engine (SnipsNLUEngine): The Snips NLU engine instance used for intent recognition.
    """

    def __init__(self, dataset_path: str = None, config: str = CONFIG_EN):
        ...

    def recognize_intent(self, text: str) -> dict:
        """
        Parses the given text and determines the intent using the trained SnipsNLUEngine.

        Args:
            text (str): Input string for which the intent is to be recognized.

        Returns:
            dict: A dictionary with detected intent and associated slots.
        """
        ...

    def get_intent_name(self, intent: dict, threshold: float = 0.5) -> str:
        """
        Extracts the intent name from the provided intent dictionary, if the probability is above the given threshold.

        Args:
            intent (dict): Recognized intent dictionary.
            threshold (float, optional): Minimum probability for considering the intent. Default is 0.5.

        Returns:
            str: Intent name if valid, otherwise None.
        """
        ...

    def get_mentioned_entities(self, intent: dict) -> dict:
        """
        Retrieves entities mentioned in the recognized intent.

        Args:
            intent (dict): Recognized intent dictionary.

        Returns:
            dict: Dictionary of entities mentioned in the intent. Returns None if no entities are found.
        """
        ...

    def is_addressed_to_drone(self, intent: dict, name: str = 'sammy', similar_pronunciation: list = None) -> bool:
        """
        Determines if the recognized intent is specifically addressed to the drone.

        Args:
            intent (dict): Recognized intent dictionary.
            name (str, optional): Name of the drone to check against. Default is 'sammy'.
            similar_pronunciation (list, optional): List of names that sound similar to the drone's name.

        Returns:
            bool: True if the intent is addressed to the drone, otherwise False.
        """
        ...

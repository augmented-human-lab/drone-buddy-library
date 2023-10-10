from dronebuddylib.atoms.intentrecognition.gpt_intent_recogntion import GPTIntentRecognition
from dronebuddylib.atoms.intentrecognition.offline_intent_recognition import OfflineIntentRecognitionEngine
from dronebuddylib.utils.enums import IntentRecognitionAlgorithm


class IntentRecognitionEngine:
    """
    A high-level engine for intent recognition leveraging various algorithms.

    Attributes:
        intent_recognizer: An instance of the intent recognition algorithm chosen.

    Args:
        algorithm (IntentRecognitionAlgorithm): The algorithm to be used for intent recognition.
        config (dict): Configuration parameters required for the chosen algorithm.

    Examples:
        >>> engine = IntentRecognitionEngine(IntentRecognitionAlgorithm.CHAT_GPT, config)
        >>> intent = engine.recognize_intent("Turn off the lights.")
    """

    def __init__(self, algorithm: IntentRecognitionAlgorithm, config):
        """
        Initialize the IntentRecognitionEngine with a given algorithm and configuration.

        Args:
            algorithm (IntentRecognitionAlgorithm): The algorithm to be used for intent recognition.
            config (dict): Configuration parameters required for the chosen algorithm.
        """
        if algorithm == IntentRecognitionAlgorithm.CHAT_GPT:
            self.intent_recognizer = GPTIntentRecognition(config)
        if algorithm == IntentRecognitionAlgorithm.SNIPS_NLU:
            self.intent_recognizer = OfflineIntentRecognitionEngine(config)

    def recognize_intent(self, text: str) -> str:
        """
        Recognize the intent from the provided text using the configured algorithm.

        Args:
            text (str): The input text from which intent needs to be recognized.

        Returns:
            str: Recognized intent.

        Examples:
            >>> intent = engine.recognize_intent("What's the weather today?")
            "get_weather"
        """
        return self.intent_recognizer.recognize_intent(text)

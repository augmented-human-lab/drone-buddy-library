from dronebuddylib.atoms.gpt_integration import GPTEngine
from dronebuddylib.models.gpt_configs import GPTConfigs
from dronebuddylib.utils.chat_prompts import SYSTEM_PROMPT_INTENT_CLASSIFICATION
from dronebuddylib.utils.utils import create_custom_drone_action_list, create_system_drone_action_list


class GPTIntentRecognition:
    """
    GPT-based intent recognition system specifically tailored for drone actions.

    This class interfaces with the GPTEngine to recognize intents from user messages,
    taking into account system-defined and custom-defined drone actions.

    Attributes:
        configs (GPTConfigs): Configurations for the GPT Engine.
        gpt_engine (GPTEngine): The GPT engine instance used for intent recognition.

    Examples:
        >>> recognizer = GPTIntentRecognition(GPTConfigs)
        >>> intent = recognizer.recognize_intent("Can you make the drone spin?")
    """

    def __init__(self, configs: GPTConfigs):
        """
        Initializes the GPTIntentRecognition with configurations and sets up the default system prompt.

        Args:
            configs (GPTConfigs): Configurations for the GPT Engine.
        """
        self.configs = configs
        self.gpt_engine = GPTEngine(configs)
        drone_actions = create_system_drone_action_list()
        modified_prompt = SYSTEM_PROMPT_INTENT_CLASSIFICATION.replace("#list", "\'" + drone_actions + "\'")
        self.gpt_engine.set_system_prompt(modified_prompt)

    def set_custom_actions_to_system_prompt(self, custom_actions: list):
        """
        Updates the system prompt with custom drone actions.

        Args:
            custom_actions (list): List of custom drone actions.
        """
        drone_actions = create_custom_drone_action_list(custom_actions)
        modified_prompt = SYSTEM_PROMPT_INTENT_CLASSIFICATION.replace("#list", "\'" + drone_actions + "\'")
        self.gpt_engine.set_system_prompt(modified_prompt)

    def get_system_prompt(self) -> str:
        """
        Retrieves the current system prompt being used.

        Returns:
            str: Current system prompt.
        """
        return SYSTEM_PROMPT_INTENT_CLASSIFICATION

    def override_system_prompt(self, system_prompt: str):
        """
        Overrides the current system prompt with the provided one.

        Args:
            system_prompt (str): The new system prompt.
        """
        self.gpt_engine.set_system_prompt(system_prompt)

    def recognize_intent(self, user_message: str) -> str:
        """
        Recognizes the intent from the provided user message using the ChatGPT engine.

        Args:
            user_message (str): The user's input message for which the intent is to be recognized.

        Returns:
            str: Recognized intent based on the user message.
        """
        return self.gpt_engine.session.get_chatgpt_response(user_message)

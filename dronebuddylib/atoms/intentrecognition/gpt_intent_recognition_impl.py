from dronebuddylib.atoms.gpt_integration import GPTEngine
from dronebuddylib.atoms.intentrecognition.i_intent_recognition import IIntentRecognition
from dronebuddylib.models.enums import Configurations
from dronebuddylib.models.gpt_configs import GPTConfigs
from dronebuddylib.utils.chat_prompts import SYSTEM_PROMPT_INTENT_CLASSIFICATION
from dronebuddylib.utils.utils import create_custom_drone_action_list, create_system_drone_action_list, \
    config_validity_check
from dronebuddylib.models.engine_configurations import EngineConfigurations


class GPTIntentRecognitionImpl(IIntentRecognition):
    """
    GPT-based intent recognition system specifically tailored for drone actions.

    This class interfaces with the GPTEngine to recognize intents from user messages,
    taking into account system-defined and custom-defined drone actions.

    Attributes:
        configs (GPTConfigs): Configurations for the GPT Engine.
        gpt_engine (GPTEngine): The GPT engine instance used for intent recognition.

    Examples:
        >>> recognizer = GPTIntentRecognitionImpl(engine_configurations)
        >>> intent = recognizer.get_resolved_intent("Can you make the drone spin?")
    """

    def get_class_name(self) -> str:
        return 'INTENT_RECOGNITION_GPT'

    def get_algorithm_name(self) -> str:
        return 'GPT Intent Recognition'

    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Initializes the GPTIntentRecognition with configurations and sets up the default system prompt.

        Args:
            configs (GPTConfigs): Configurations for the GPT Engine.
        """

        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())
        configs = engine_configurations.get_configurations_for_engine(self.get_class_name())
        self.configs = engine_configurations

        gpt_configs = GPTConfigs(
            configs.get(Configurations.INTENT_RECOGNITION_OPEN_AI_API_KEY.name,
                        configs.get(Configurations.INTENT_RECOGNITION_OPEN_AI_API_KEY)),
            configs.get(Configurations.INTENT_RECOGNITION_OPEN_AI_MODEL.name,
                        configs.get(Configurations.INTENT_RECOGNITION_OPEN_AI_MODEL)),
            configs.get(Configurations.INTENT_RECOGNITION_OPEN_AI_TEMPERATURE.name,
                        configs.get(Configurations.INTENT_RECOGNITION_OPEN_AI_TEMPERATURE)),
            configs.get(Configurations.INTENT_RECOGNITION_OPEN_AI_API_URL.name,
                        configs.get(Configurations.INTENT_RECOGNITION_OPEN_AI_API_URL)),
            configs.get(Configurations.INTENT_RECOGNITION_OPEN_AI_LOGGER_LOCATION.name,
                        configs.get(Configurations.INTENT_RECOGNITION_OPEN_AI_LOGGER_LOCATION))
        )

        self.gpt_engine = GPTEngine(gpt_configs)

        if configs.get(Configurations.INTENT_RECOGNITION_SYSTEM_PROMPT.name, None) is not None:
            self.override_system_prompt(configs.get(Configurations.INTENT_RECOGNITION_SYSTEM_PROMPT.name))
        else:
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

    def get_resolved_intent(self, user_message: str) -> str:
        """
        Recognizes the intent from the provided user message using the ChatGPT engine.

        Args:
            user_message (str): The user's input message for which the intent is to be recognized.

        Returns:
            str: Recognized intent based on the user message.
        """
        return self.gpt_engine.session.get_chatgpt_response(user_message)

    def get_required_params(self) -> list:
        return [Configurations.INTENT_RECOGNITION_OPEN_AI_API_KEY,
                Configurations.INTENT_RECOGNITION_OPEN_AI_API_URL,
                Configurations.INTENT_RECOGNITION_OPEN_AI_MODEL,
                Configurations.INTENT_RECOGNITION_OPEN_AI_TEMPERATURE,
                Configurations.INTENT_RECOGNITION_OPEN_AI_LOGGER_LOCATION,
                ]

    def get_optional_params(self) -> list:
        return [Configurations.INTENT_RECOGNITION_SYSTEM_PROMPT]

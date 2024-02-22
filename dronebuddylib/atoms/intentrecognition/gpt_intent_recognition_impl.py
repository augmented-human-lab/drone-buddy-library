import json

from dronebuddylib.atoms.gpt_integration import GPTEngine
from dronebuddylib.atoms.intentrecognition.i_intent_recognition import IIntentRecognition
from dronebuddylib.exceptions.intent_resolution_exception import IntentResolutionException
from dronebuddylib.models.enums import AtomicEngineConfigurations
from dronebuddylib.models.gpt_configs import GPTConfigs
from dronebuddylib.atoms.intentrecognition.recognized_intent_result import RecognizedIntent, RecognizedEntities
from dronebuddylib.utils.chat_prompts import SYSTEM_PROMPT_INTENT_CLASSIFICATION
from dronebuddylib.utils.utils import create_custom_drone_action_list, create_system_drone_action_list, \
    config_validity_check, logger
from dronebuddylib.models.engine_configurations import EngineConfigurations


class GPTIntentRecognitionImpl(IIntentRecognition):
    """
    GPT-based intent recognition system specifically tailored for drone actions.

    This class interfaces with the GPTEngine to recognize intents from user messages,
    taking into account system-defined and custom-defined drone actions.

    Attributes:
        configs (GPTConfigs): Configurations for the GPT Engine.
        gpt_engine (GPTEngine): The GPT engine instance used for intent recognition.
    """

    def introduce_new_intents(self, new_intents: dict) -> bool:
        self.set_custom_actions_to_system_prompt(SYSTEM_PROMPT_INTENT_CLASSIFICATION, new_intents)
        return True

    def get_class_name(self) -> str:
        """
        Returns the class name of the intent recognition implementation.

        Returns:
            str: The class name.
        """
        return 'INTENT_RECOGNITION_OPEN_AI'

    def get_algorithm_name(self) -> str:
        """
        Returns the algorithm name of the intent recognition implementation.

        Returns:
            str: The algorithm name.
        """
        return 'GPT Intent Recognition'

    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Initializes the GPTIntentRecognition with configurations and sets up the default system prompt.

        Args:
            engine_configurations (EngineConfigurations): Configurations for the GPT Engine.
        """
        # Validate configurations
        super().__init__(engine_configurations)
        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())

        # Get configurations
        configs = engine_configurations.get_configurations_for_engine(self.get_class_name())
        self.configs = engine_configurations

        # Initialize GPT configs
        gpt_configs = GPTConfigs(
            open_ai_api_key=configs.get(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_API_KEY.name,
                                        configs.get(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_API_KEY)),
            open_ai_model=configs.get(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_MODEL.name,
                                      configs.get(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_MODEL)),
            open_ai_temperature=configs.get(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_TEMPERATURE.name,
                                            configs.get(
                                                AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_TEMPERATURE)),
            open_ai_api_url=configs.get(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_API_URL.name,
                                        configs.get(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_API_URL)),
            loger_location=configs.get(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_LOGGER_LOCATION.name,
                                       configs.get(
                                           AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_LOGGER_LOCATION))
        )

        overriden_system_prompt = configs.get(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_SYSTEM_PROMPT.name,
                                              configs.get(
                                                  AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_SYSTEM_PROMPT))
        overriden_list_path = configs.get(
            AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_SYSTEM_ACTIONS_PATH.name,
            configs.get(
                AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_SYSTEM_ACTIONS_PATH))

        # Initialize GPT engine
        self.gpt_engine = GPTEngine(gpt_configs)

        system_action_string = ""
        overriden_list = {}

        if overriden_list_path is not None:
            # read the actions from the file and add it to a dictionary
            #  the file has the format action=value
            #  the file is at overriden_list_path
            with open(overriden_list_path, 'r') as file:
                for line in file:
                    action, value = line.split('=')
                    overriden_list[action] = value

        if overriden_system_prompt is not None:
            system_action_string = create_custom_drone_action_list(overriden_list)
        else:
            system_action_string = create_system_drone_action_list()

        current_system_prompt = ""

        # Set system prompt
        if configs.get(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_SYSTEM_PROMPT.name, None) is not None:
            current_system_prompt = configs.get(
                AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_SYSTEM_PROMPT.name)
        else:
            current_system_prompt = SYSTEM_PROMPT_INTENT_CLASSIFICATION

        modified_prompt = current_system_prompt.replace("#list", "\'" + system_action_string + "\'")

        self.gpt_engine.set_system_prompt(modified_prompt)

        logger.log_debug(self.get_class_name(), 'Completed initializing the GPT intent recognition')

    def set_custom_actions_to_system_prompt(self, prompt: str, custom_actions: dict):
        """
        Updates the system prompt with custom drone actions.

        Args:
            custom_actions (list): List of custom drone actions.
        """
        drone_actions = create_custom_drone_action_list(custom_actions)
        modified_prompt = prompt.replace("#list", "\'" + drone_actions + "\'")
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

    def get_resolved_intent(self, user_message: str) -> RecognizedIntent:
        """
        Recognizes the intent from the provided user message using the ChatGPT engine.

        Args:
            user_message (str): The user's input message for which the intent is to be recognized.

        Returns:
            str: Recognized intent based on the user message.
        """
        try:
            logger.log_debug(self.get_class_name(), ' Recognition started.')

            result = self.gpt_engine.session.get_chatgpt_response(user_message)
            logger.log_debug(self.get_class_name(), ' Recognition successful.')

            json_result = json.loads(result)
            addressee = None
            try:
                addressee = json_result['addressed_to']
            except KeyError:
                addressee = ""
            confidence = None
            try:
                confidence = json_result['confidence']
            except KeyError:
                confidence = 0

            formatted_result = RecognizedIntent(json_result['intent'], [], confidence,

                                                addressee)
            try:
                if json_result['entities'] is not None:
                    for entity in json_result['entities']:
                        formatted_result.entities.append(RecognizedEntities(entity['entity_type'], entity['value']))
                else:
                    formatted_result.entities = []
            except KeyError:
                pass
            logger.log_debug(self.get_class_name(), ' Recognition completed.')

            return formatted_result
        except KeyError:
            raise IntentResolutionException("Intent could not be resolved.", 500)

    def get_required_params(self) -> list:
        """
        Returns the list of required configuration parameters for the intent recognition engine.

        Returns:
            list: List of required configuration parameters.
        """
        return [AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_API_KEY,
                AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_API_URL,
                AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_MODEL,
                AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_TEMPERATURE,
                AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_LOGGER_LOCATION]

    def get_optional_params(self) -> list:
        """
        Returns the list of optional configuration parameters for the intent recognition engine.

        Returns:
            list: List of optional configuration parameters.
        """
        return [AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_SYSTEM_PROMPT,
                AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_SYSTEM_ACTIONS_PATH]

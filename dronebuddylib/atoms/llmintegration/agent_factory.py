from dronebuddylib.atoms.llmintegration.image_describer_agent_impl import ImageDescriberAgentImpl
from dronebuddylib.atoms.llmintegration.image_validator_agent_impl import ImageValidatorAgentImpl
from dronebuddylib.atoms.llmintegration.intent_resolver_agent_impl import IntentResolverAgentImpl
from dronebuddylib.atoms.llmintegration.object_identifier_agent_impl import ObjectIdentifierAgentImpl
from dronebuddylib.utils.enums import LLMAgentNames


class AgentFactory:

    def __init__(self, model_name: str, api_key: str, temperature: float = None):
        """
        Initialize the AgentPickerEngine with the given openai model name, API key and temperature
        :param model_name: The name of the OpenAI model to use
        :param api_key: The API key to use
        :param temperature: The temperature to use
        """
        self.object_identifier_agent = ObjectIdentifierAgentImpl(api_key, model_name, temperature)
        self.image_describer_agent = ImageDescriberAgentImpl(api_key, model_name, temperature)
        self.intent_resolver_agent = IntentResolverAgentImpl(api_key, model_name, temperature)
        self.image_validator_agent = ImageValidatorAgentImpl(api_key, model_name, temperature)

    def get_agent(self, agent: LLMAgentNames):
        if agent == LLMAgentNames.OBJECT_IDENTIFIER:
            return self.object_identifier_agent
        elif agent == LLMAgentNames.IMAGE_DESCRIBER:
            return self.image_describer_agent
        elif agent == LLMAgentNames.INTENT_RESOLVER:
            return self.intent_resolver_agent
        elif agent == LLMAgentNames.IMAGE_VALIDATOR:
            return self.image_validator_agent
        else:
            return None

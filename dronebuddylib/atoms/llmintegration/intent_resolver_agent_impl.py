from dronebuddylib.atoms.llmintegration.i_llm_agent import ILLMAgent
from dronebuddylib.utils.enums import LLMAgentNames


class IntentResolverAgentImpl(ILLMAgent):
    SYSTEM_PROMPT_IMAGE_DESCRIBER = """
    You are a helpful assistant.
    """

    def __init__(self, api_key: str, model_name: str, temperature: float = None, logger_location: str = None):
        super().__init__(api_key, model_name, temperature, logger_location)
        self.set_system_prompt(self.SYSTEM_PROMPT_IMAGE_DESCRIBER)

    def get_agent_name(self):
        return LLMAgentNames.IMAGE_DESCRIBER.name

    def get_agent_description(self):
        return LLMAgentNames.IMAGE_DESCRIBER.value

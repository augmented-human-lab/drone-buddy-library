import pkg_resources
from openai.types.chat import ChatCompletionMessage

from dronebuddylib.models.chat_session import ChatSession
from dronebuddylib.models.gpt_configs import GPTConfigs
from dronebuddylib.utils.enums import LLMAgentNames


class ILLMAgent:
    def __init__(self, api_key: str, model_name: str, temperature: float = None, logger_location: str = None):
        self.api_key = api_key
        if logger_location is None:
            logger_location = pkg_resources.resource_filename(__name__, "resources/statistics/agent_data")
        self.model_name = model_name

        self.llm_configs = GPTConfigs(
            open_ai_api_key=api_key,
            open_ai_model=model_name,
            open_ai_temperature=temperature,
            loger_location=logger_location
        )

        # Initialize LLM engine
        self.llm_session = ChatSession(self.llm_configs)

    def get_llm_session(self) -> ChatSession:
        return self.llm_session

    def get_agent_name(self) -> LLMAgentNames:
        pass

    def set_system_prompt(self, system_prompt):
        self.llm_session.set_system_prompt(system_prompt)

    def get_agent_description(self) -> LLMAgentNames:
        pass

    def send_text_message_to_llm_queue(self, role, content):
        self.llm_session.send_text_message_to_llm(role, content)

    def send_image_message_to_llm_queue(self, role, content, image_path):
        self.llm_session.send_image_message_to_llm(role, content, image_path)

    def send_encoded_image_message_to_llm_queue(self, role, content, image):
        self.llm_session.send_encoded_image_message_to_llm_queue(role, content, image)

    def get_response_from_llm(self) -> ChatCompletionMessage:
        return self.llm_session.get_response()

    def get_result(self):
        pass

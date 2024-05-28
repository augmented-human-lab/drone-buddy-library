from dronebuddylib.models.chat_session import ChatSession
from dronebuddylib.models.gpt_configs import GPTConfigs


class GPTEngine:
    def __init__(self, configs: GPTConfigs):
        session = ChatSession(configs)
        self.session = session

    def set_system_prompt(self, system_prompt: str):
        self.session.set_system_prompt(system_prompt)

    def get_response(self, user_message: str) -> str:
        """
        For the given user_message,
        get the response from ChatGPT
        """
        return self.session.get_chatgpt_response(user_message)

    def get_response_for_image_queries(self, user_message: str, image_path: str) -> str:
        """
        For the given user_message,
        get the response from ChatGPT
        """
        return self.session.get_chatgpt_response_for_image_queries(user_message, image_path)
    def get_response_for_image_queries_as_files(self, user_message: str, image_path: str) -> str:
        """
        For the given user_message,
        get the response from ChatGPT
        """
        return self.session.get_chatgpt_response_for_image_queries(user_message, image_path)

    def add_message_with_image(self, role, content, image_path):
        self.session.conversation.add_image_message(role, content, image_path)

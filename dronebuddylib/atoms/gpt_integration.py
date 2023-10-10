from dronebuddylib.models.chat_session import ChatSession
from dronebuddylib.models.gpt_configs import GPTConfigs


class GPTEngine:
    def __int__(self, configs: GPTConfigs):
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

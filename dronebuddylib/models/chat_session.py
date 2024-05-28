import base64
import uuid
from typing import List, Dict

import cv2
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from dronebuddylib.exceptions.intent_resolution_exception import IntentResolutionException
from dronebuddylib.models.conversation import Conversation
from dronebuddylib.models.gpt_configs import GPTConfigs
from dronebuddylib.models.session_logger import SessionLogger
from dronebuddylib.models.token_counter import num_tokens_from_messages


class ChatSession:
    """
    Represents a chat session.
    Each session has a unique id to associate it with the user.
    It holds the conversation history
    and provides functionality to get new response from ChatGPT
    for user query.
    """

    def __init__(self, configs: GPTConfigs):
        self.session_id = str(uuid.uuid4())
        self.conversation = Conversation()

        # get action list from the enum class as a list
        self.openai_model = configs.open_ai_model
        self.openai_temperature = configs.open_ai_temperature
        self.logger = SessionLogger(configs.loger_location)
        self.openai = OpenAI(api_key=configs.open_ai_api_key)

    def set_system_prompt(self, system_prompt: str):
        self.conversation.add_message("system", system_prompt)

    def get_messages(self) -> List[Dict]:
        """
        Return the list of messages from the current conversation
        """
        # Exclude the SYSTEM_PROMPT when returning the history
        if len(self.conversation.conversation_history) == 1:
            return []
        return self.conversation.conversation_history[1:]

    def get_response(self) -> ChatCompletionMessage:
        return self._chat_completion_request(
            self.conversation.conversation_history
        )

    def get_chatgpt_response(self, user_message: str) -> str:
        """
        For the given user_message,
        get the response from ChatGPT
        """
        self.conversation.add_message("user", user_message)
        token_count = num_tokens_from_messages(self.conversation.conversation_history, self.openai_model)
        self.logger.log_chat('user', token_count, user_message)
        try:
            chatgpt_response = self._chat_completion_request(
                self.conversation.conversation_history
            )
            chatgpt_message = chatgpt_response.content
            self.conversation.add_message("assistant", chatgpt_message)
            self.logger.log_chat('user', -1, chatgpt_message)

            return chatgpt_message
        except Exception as e:
            print(e)
            raise IntentResolutionException("Intent could not be resolved.", 500)

    def send_text_message_to_llm(self, role, content):
        self.conversation.add_message(role, content)

    def send_image_message_to_llm(self, role, content, image_path):
        self.conversation.add_image_message(role, content,image_path)

    def send_encoded_image_message_to_llm_queue(self, role, content, image):
        self.conversation.add_image_message_as_encoded(role, content, image)

    def get_chatgpt_response_for_image_queries(self, user_message: str, image_path: str) -> str:
        """
        For the given user_message,
        get the response from ChatGPT
        """
        self.conversation.add_image_message("user", user_message, image_path)
        # token_count = num_tokens_from_messages(self.conversation.conversation_history, self.openai_model)
        # self.logger.log_chat('user', token_count, user_message)
        try:
            chatgpt_response = self._chat_completion_request(
                self.conversation.conversation_history
            )
            chatgpt_message = chatgpt_response.content
            self.conversation.add_message("assistant", chatgpt_message)
            self.logger.log_chat('assistant', -1, chatgpt_message)

            return chatgpt_message
        except Exception as e:
            print(e)
            raise IntentResolutionException("Intent could not be resolved.", 500)

    def get_chatgpt_response_for_image_queries_as_encoded(self, user_message: str, image_path: str) -> str:
        """
        For the given user_message,
        get the response from ChatGPT
        """
        self.conversation.add_image_message("user", user_message, image_path)
        # token_count = num_tokens_from_messages(self.conversation.conversation_history, self.openai_model)
        # self.logger.log_chat('user', token_count, user_message)
        try:
            chatgpt_response = self._chat_completion_request(
                self.conversation.conversation_history
            )
            chatgpt_message = chatgpt_response.content
            self.conversation.add_message("assistant", chatgpt_message)
            self.logger.log_chat('assistant', -1, chatgpt_message)

            return chatgpt_message
        except Exception as e:
            print(e)
            raise IntentResolutionException("Intent could not be resolved.", 500)

    def encode_image_cv(self, image) -> str:
        """
        Encode an OpenCV image to a base64 string
        """
        return base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8')

    def send_message_with_image(self, user_message: str, image_path: str) -> str:
        """
        Send a message with an attached image
        """
        # base64_image = self.encode_image(image_path)
        self.conversation.add_image_message("user", user_message, image_path)
        return self.get_chatgpt_response_for_image_queries(user_message, image_path)

    def _chat_completion_request(self, messages: List[Dict]):
        completion = self.openai.chat.completions.create(
            model=self.openai_model,
            messages=messages,
        )
        return completion.choices[0].message

    def end_session(self):
        self.logger.close_file()

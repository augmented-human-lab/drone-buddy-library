from typing import List, Dict


class Conversation:
    """
        This class represents a conversation with the ChatGPT model.
        It stores the conversation history in the form of a list of
        messages.
        """

    def __init__(self):
        self.conversation_history: List[Dict] = []

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.conversation_history.append(message)

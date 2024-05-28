import base64
from typing import List, Dict

import cv2


class Conversation:
    """
        This class represents a conversation with the ChatGPT model.
        It stores the conversation history in the form of a list of
        messages.
        """

    def __init__(self):
        self.conversation_history: List[Dict] = []

    def add_message(self, role, content):
        message = {
            "role": role,
            "content": [{
                "type": "text",
                "text": content, }
            ]}
        self.conversation_history.append(message)

    def add_image_message(self, role, content, image_url):
        message = {
            "role": role,
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": content
                }
            ]
        }
        self.conversation_history.append(message)

    def encode_image(self, image_path: str) -> str:
        """
        Encode the image at the given path to a base64 string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_base64_encoded_image(self, frame):

        # Encode frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)

        if ret:
            # Convert to base64
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return image_base64
        else:
            print("Failed to encode image")
            return None

    def add_image_message_as_encoded(self, role, content, image):
        encoded_image = self.get_base64_encoded_image(image)
        message = {
            "role": role,
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                },
                {
                    "type": "text",
                    "text": content
                }
            ]
        }
        self.conversation_history.append(message)

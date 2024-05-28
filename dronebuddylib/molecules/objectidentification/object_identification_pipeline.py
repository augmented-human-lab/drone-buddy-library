import cv2
import numpy as np

from dronebuddylib.atoms.llmintegration.agent_picker_engine import AgentPickerEngine
from dronebuddylib.atoms.llmintegration.i_llm_agent import ILLMAgent
from dronebuddylib.utils.enums import LLMAgentNames


def get_objects_in_image(image: np.ndarray):
    identified_objects = []
    available_objects = []
    # return IdentifiedObjects(identified_objects, available_objects)


def describe_the_retrieved_image(image, image_describer_agent: ILLMAgent):
    image_describer_agent.send_encoded_image_message_to_llm_queue("user", "DESCRIBE", image)
    return image_describer_agent.get_result()


def test_object_identification_pipeline():
    model = "gpt-4o"
    openai_api_key = "sk-proj-b4Xiugvz43TNjOvvbR6aT3BlbkFJOYuEW5pjBghRK4bAqq80"

    agent_picker = AgentPickerEngine(model, openai_api_key, None)
    # Initialize GPT engine
    image_describer = agent_picker.get_agent(LLMAgentNames.IMAGE_DESCRIBER)
    image = cv2.imread(r'C:\Users\Public\projects\drone-buddy-library\test\object_images\hot_bottle.jpeg')
    result = describe_the_retrieved_image(image, image_describer)
    print(result)

def get_confirmation_from_user():
    pass


def send_image_to_memory(image):
    pass


def validate_image(image_validator_agent: ILLMAgent, image, object_name):
    image_validator_agent.send_encoded_image_message_to_llm_queue("user", "VALIDATE(" + object_name + ")", image)
    result = image_validator_agent.get_result()
    print(result)
    return result


# add main method
if __name__ == "__main__":
    test_object_identification_pipeline()

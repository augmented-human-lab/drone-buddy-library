import json

from dronebuddylib.atoms.llmintegration.i_llm_agent import ILLMAgent
from dronebuddylib.atoms.llmintegration.models.image_describer_results import ImageDescriberResults
from dronebuddylib.utils.enums import LLMAgentNames


class ImageDescriberAgentImpl(ILLMAgent):
    """
    A class to implement an image describer agent using an LLM (Large Language Model).
    This class provides functionalities to describe images to help visually impaired individuals.
    """
    SYSTEM_PROMPT_IMAGE_DESCRIBER = """
    You are a helpful assistant helping a visually impaired person navigate in their day to day life.

    When an image and the instruction DESCRIBE is given, explain the object as much as possible and only explain the object.

    Give the result in the format

    {
        "object_name": "what kind of object it is",
        "description": "what the object is, a full description for the person to listen to",
        "confidence": a numerical value for the confidence
    }
    """

    def __init__(self, api_key: str, model_name: str, temperature: float = None, logger_location: str = None):
        """
        Initializes the ImageDescriberAgentImpl with the given parameters.

        Args:
            api_key (str): The API key for accessing the LLM.
            model_name (str): The name of the model to be used.
            temperature (float, optional): The temperature setting for the model's responses.
            logger_location (str, optional): The location for logging information.
        """
        super().__init__(api_key, model_name, temperature, logger_location)
        self.set_system_prompt(self.SYSTEM_PROMPT_IMAGE_DESCRIBER)

    def get_agent_name(self):
        """
        Gets the name of the LLM agent.

        Returns:
            str: The name of the LLM agent.
        """
        return LLMAgentNames.IMAGE_DESCRIBER.name

    def get_agent_description(self):
        """
        Gets the description of the LLM agent.

        Returns:
            str: The description of the LLM agent.
        """
        return LLMAgentNames.IMAGE_DESCRIBER.value

    def get_result(self) -> ImageDescriberResults:
        """
        Gets the description result from the LLM and formats it into an ImageDescriberResults object.

        Returns:
            ImageDescriberResults: The formatted result of the image description.
        """
        result = self.get_response_from_llm().content
        formatted_result = json.loads(result)
        description = ImageDescriberResults(formatted_result['object_name'], formatted_result['description'],
                                            formatted_result['confidence'])

        return description

import json

from dronebuddylib.atoms.llmintegration.i_llm_agent import ILLMAgent
from dronebuddylib.atoms.llmintegration.models.image_validator_results import ImageValidatorResults
from dronebuddylib.utils.enums import LLMAgentNames


class ImageValidatorAgentImpl(ILLMAgent):
    """
    A class to implement an image validator agent using an LLM (Large Language Model).
    This class provides functionalities to validate images for object identification.
    """
    SYSTEM_PROMPT_IMAGE_VALIDATOR = """
    You are a helpful assistant, capable of deciding whether a given image is suitable for remembering to be recognized in future cases. The image should be clear, and have the full object that need to be focused.

    When given the image, with the instruction VALIDATE(type of the object),

    return the result in the format

    { 
    "data":
        [
             { 
                "object_type": "type of the object", 

                "is_valid": "true if the image is good enough/ false if not", "description": "description of the object", 
                "instructions": 
                            if the area of the image needs to be higher - INCREASE,

                            if the lighting needs to be improved - LIGHTING_IMPROVE,

                            if the object is incomplete : INCOMPLETE, 

                            if the object is not in focus : NOT_FOCUSED
            } 
        ] 
    }
    """

    def __init__(self, api_key: str, model_name: str, temperature: float = None, logger_location: str = None):
        """
        Initializes the ImageValidatorAgentImpl with the given parameters.

        Args:
            api_key (str): The API key for accessing the LLM.
            model_name (str): The name of the model to be used.
            temperature (float, optional): The temperature setting for the model's responses.
            logger_location (str, optional): The location for logging information.
        """
        super().__init__(api_key, model_name, temperature, logger_location)
        self.set_system_prompt(self.SYSTEM_PROMPT_IMAGE_VALIDATOR)

    def get_agent_name(self):
        """
        Gets the name of the LLM agent.

        Returns:
            str: The name of the LLM agent.
        """
        return LLMAgentNames.IMAGE_VALIDATOR.name

    def get_agent_description(self):
        """
        Gets the description of the LLM agent.

        Returns:
            str: The description of the LLM agent.
        """
        return LLMAgentNames.IMAGE_VALIDATOR.value

    def get_result(self) -> ImageValidatorResults:
        """
        Gets the validation result from the LLM and formats it into an ImageValidatorResults object.

        Returns:
            ImageValidatorResults: The formatted result of the image validation.
        """
        result = self.get_response_from_llm().content
        formatted_result = json.loads(result)
        description = ImageValidatorResults(formatted_result['object_type'], formatted_result['is_valid'],
                                            formatted_result['description'], formatted_result['instructions'])

        return description

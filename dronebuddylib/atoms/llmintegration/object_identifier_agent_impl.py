from dronebuddylib.atoms.llmintegration.i_llm_agent import ILLMAgent
from dronebuddylib.utils.enums import LLMAgentNames


class ObjectIdentifierAgentImpl(ILLMAgent):
    """
    A class to implement an object identifier agent using an LLM (Large Language Model).
    This class provides functionalities to remember objects and identify objects in images.
    """
    SYSTEM_PROMPT_OBJECT_IDENTIFICATION = """
    You are a helpful assistant.

    When the instruction "REMEMBER_AS(object name)" is given with an image of the object, 
    remember the object and return an acknowledgement in the format of:

    {
        "status": "SUCCESS" (if successfully added to the memory) / "UNSUCCESSFUL" (if otherwise),
        "message": "description"
    }

    Once the instruction "IDENTIFY" is given with the image, 
    return all the identified objects in the form of a JSON object:
    {
        "data": [
            {
                "class_name": "class the object belongs to",
                "object_name": "name of the remembered object / unknown if not a previously remembered object",
                "description": "description of the object",
                "confidence": confidence as a value
            }
        ]
    }
    """

    def __init__(self, api_key: str, model_name: str, temperature: float = None, logger_location: str = None):
        """
        Initializes the ObjectIdentifierAgentImpl with the given parameters.

        Args:
            api_key (str): The API key for accessing the LLM.
            model_name (str): The name of the model to be used.
            temperature (float, optional): The temperature setting for the model's responses.
            logger_location (str, optional): The location for logging information.
        """
        super().__init__(api_key, model_name, temperature, logger_location)
        self.set_system_prompt(self.SYSTEM_PROMPT_OBJECT_IDENTIFICATION)

    def get_agent_name(self):
        """
        Gets the name of the LLM agent.

        Returns:
            str: The name of the LLM agent.
        """
        return LLMAgentNames.OBJECT_IDENTIFIER.name

    def get_agent_description(self):
        """
        Gets the description of the LLM agent.

        Returns:
            str: The description of the LLM agent.
        """
        return LLMAgentNames.OBJECT_IDENTIFIER.value

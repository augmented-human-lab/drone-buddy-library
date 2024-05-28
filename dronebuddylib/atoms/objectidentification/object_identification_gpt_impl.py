import json
import os
import pickle
import re
import threading
import time
from asyncio import Future
from pathlib import Path

import cv2
import openai
import pkg_resources
from openai import OpenAI

from tqdm import tqdm

from dronebuddylib.atoms.llmintegration.agent_factory import AgentFactory
from dronebuddylib.atoms.llmintegration.models.image_validator_results import ImageValidatorResults
from dronebuddylib.atoms.objectidentification.i_object_identification import IObjectIdentification
from dronebuddylib.atoms.objectidentification.object_identification_result import IdentifiedObjects, \
    IdentifiedObjectObject
from dronebuddylib.exceptions.llm_exception import LLMException
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import AtomicEngineConfigurations
from dronebuddylib.utils.enums import LLMAgentNames
from dronebuddylib.utils.logger import Logger

logger = Logger()


class ObjectIdentificationGPTImpl(IObjectIdentification):
    """
    A class to perform object identification using ResNet and GPT integration.
    """
    progress_event = threading.Event()

    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Initializes the ResNet object detection engine with the given engine configurations.

        Args:
            engine_configurations (EngineConfigurations): The engine configurations for the object detection engine.
        """
        super().__init__(engine_configurations)

        configs = engine_configurations.get_configurations_for_engine(self.get_class_name())

        openai_api_key = configs.get(AtomicEngineConfigurations.OBJECT_IDENTIFICATION_GPT_API_KEY.name,
                                     configs.get(AtomicEngineConfigurations.OBJECT_IDENTIFICATION_GPT_API_KEY))
        self.model = "gpt-4o"
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=self.openai_api_key)

        self.agent_picker = AgentFactory(self.model, self.openai_api_key, None)
        self.object_identifier = self.agent_picker.get_agent(LLMAgentNames.OBJECT_IDENTIFIER)
        self.image_validator = self.agent_picker.get_agent(LLMAgentNames.IMAGE_VALIDATOR)
        self.image_describer = self.agent_picker.get_agent(LLMAgentNames.IMAGE_DESCRIBER)

    def create_memory_on_the_fly(self, changes=None):
        """
        Creates memory on the fly by reading known objects from a JSON file and
        sending them to the object identifier for processing.

        Args:
            changes: Optional parameter to pass changes if any.
        """
        file_path = pkg_resources.resource_filename(__name__, "resources/known_object_list.json")
        with open(file_path, 'r') as file:
            json_content = json.load(file)
            known_object_list = json_content['known_objects']
            for obj in known_object_list:
                self.object_identifier.send_image_message_to_llm_queue("user", "REMEMBER_AS( " + obj['name'] + ")",
                                                                       obj['url'])
                print(obj)
        print("Memory created successfully")

    def identify_object(self, image) -> IdentifiedObjects:
        """
        Identifies the objects in the given image using the ResNet object detection engine.

        Args:
            image_path (str): The path to the image of the objects to identify.

        Returns:
            IdentifiedObjects: The identified objects with their associated probabilities.
        """
        self.object_identifier.send_encoded_image_message_to_llm_queue("user", "IDENTIFY", image)
        result = self.object_identifier.get_response_from_llm().content
        formatted_result = self.format_answers(result)
        return formatted_result

    def identify_object_image_path(self, image_path) -> IdentifiedObjects:
        """
        Identifies the objects in the given image using the ResNet object detection engine.

        Args:
            image_path (str): The path to the image of the objects to identify.

        Returns:
            IdentifiedObjects: The identified objects with their associated probabilities.
        """
        self.object_identifier.send_image_message_to_llm_queue("user", "IDENTIFY", image_path)
        result = self.object_identifier.get_response_from_llm().content
        formatted_result = self.format_answers(result)
        print(result)
        return formatted_result

    def format_answers(self, result):
        """
        Formats the raw result from the object identifier into IdentifiedObjects.

        Args:
            result (str): The raw JSON result from the object identifier.

        Returns:
            IdentifiedObjects: The formatted identified objects.
        """
        formatted_result = json.loads(result)
        identified_objects = IdentifiedObjects([], [])
        for result in formatted_result['data']:
            object_name = result['object_name']
            obj = IdentifiedObjectObject(result['class_name'], result['object_name'], result['description'],
                                         result['confidence'])
            if object_name == "unknown":
                identified_objects.add_available_object(obj)
            else:
                identified_objects.add_identified_object(obj)

        return identified_objects

    def remember_object(self, image=None, type=None, name=None):
        """
        Remembers a new object by sending its image and type to the object identifier.

        Args:
            image: The image of the object to remember.
            type: The type of the object.
            name: The name of the object.

        Returns:
            success_result: The result from the object identifier after processing.
        """
        logger.log_info(self.get_class_name(), 'Starting to remember object: type : ' + type + ' : ' + name)
        validation_result = self.validate_reference_image(image, type)
        if validation_result.is_valid:
            success_result = self.object_identifier.send_encoded_image_message_to_llm_queue("user",
                                                                                            "REMEMBER_AS( " + name + ")",
                                                                                            image)
            return success_result
        else:
            logger.log_error(self.get_class_name(), 'Image validation failed. Please try again with a different image.')
            return LLMException('Image validation failed. Please try again with a different image.', 500,
                                str(validation_result))

    def validate_reference_image(self, image, image_type) -> ImageValidatorResults:
        """
        Validates the reference image using the object validator.

        Args:
            image: The image to validate.
            image_type: The type of the image.

        Returns:
            ImageValidatorResults: The result of the image validation.
        """
        validity = self.image_validator.send_encoded_image_message_to_llm_queue("user",
                                                                                "VALIDATE( " + image_type + ")",
                                                                                image)
        return validity

    def describe_image(self, frame):
        """
        Describes the image using the GPT model.

        Args:
            frame: The image to describe.

        Returns:
            str: The description of the image.
        """
        description = self.image_describer.get_response_for_image_queries("DESCRIBE", frame)
        return description

    def get_class_name(self) -> str:
        """
        Gets the class name of the object detection implementation.

        Returns:
            str: The class name of the object detection implementation.
        """
        return 'OBJECT_IDENTIFICATION_GPT'

    def get_algorithm_name(self) -> str:
        """
        Gets the algorithm name of the object detection implementation.

        Returns:
            str: The algorithm name of the object detection implementation.
        """
        return 'Chat GPT object identification'

    def get_required_params(self) -> list:
        """
        Gets the list of required configuration parameters for the GPT object detection engine.

        Returns:
            list: The list of required configuration parameters.
        """
        return [AtomicEngineConfigurations.OBJECT_IDENTIFICATION_GPT_API_KEY]

    def get_optional_params(self) -> list:
        """
        Gets the list of optional configuration parameters for the GPT object detection engine.

        Returns:
            list: The list of optional configuration parameters.
        """
        return [AtomicEngineConfigurations.OBJECT_IDENTIFICATION_GPT_MODEL]

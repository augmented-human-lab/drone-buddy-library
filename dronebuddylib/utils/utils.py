import os

import pkg_resources

from dronebuddylib.models.enums import DroneCommands
from dronebuddylib.utils.exceptions import MissingConfigurationException
from dronebuddylib.utils.logger import Logger

logger = Logger()


def create_system_drone_action_list() -> str:
    list_actions = [e.name for e in DroneCommands]
    action_string = ""
    for action in list_actions:
        action_string = action_string + action + "\n"

    return action_string


def get_current_intents() -> dict:
    text_file_path = pkg_resources.resource_filename(__name__, "intentrecognition/resources/intents.txt")

    try:
        with open(text_file_path, "r") as file:
            # Read the contents of the file line by line
            lines = file.readlines()
            lines_without_newline = [line.rstrip('\n') for line in lines]
            intent_list = [line for line in lines_without_newline if line]
            intent_dict = {}
            for intent in intent_list:
                intent_name, intent_description = intent.split("=")
                intent_dict[intent_name] = intent_description
            return intent_dict
    except FileNotFoundError as e:
        raise FileNotFoundError("The specified file is not found.", e) from e


def create_custom_drone_action_list(custom_actions: dict) -> str:
    action_string = ""
    for action in custom_actions:
        action_string = action_string + action + " = \'" + custom_actions[action] + "\'" + "\n"

    return action_string


def config_validity_check(class_requirements: list, provided_configs: dict, algo_name: str):
    if class_requirements is not None:
        if len(provided_configs) == 0 and len(class_requirements) > 0:
            logger.log_error("Utils",
                             'Missing configuration to initialize the algorithm: ' + algo_name + ' : configuration: ' + "All")
            raise MissingConfigurationException(algo_name, "All")
        if class_requirements is not None:
            for req_key in class_requirements:
                try:
                    provided_configs.pop(req_key, None)
                except KeyError:
                    logger.log_error(
                        "UTILS",
                        'Missing configuration to initialize the algorithm: ' + algo_name + ' : configuration: ' + req_key)
                    raise MissingConfigurationException(algo_name, req_key)


def write_to_file(filename, integer):
    """Write an integer to a file."""
    try:
        with open(filename, 'w') as file:  # 'w' mode will create the file if it doesn't exist
            file.write(str(integer))
            return True
    except IOError:
        logger.log_error("Utils", "Error while writing to the file : " + filename)
        return False


def write_to_file_longer(filename, integer):
    """Write an integer to a file."""
    """Appends an integer to a file on a new line. For the first entry, it does not prepend a newline."""
    try:
        # Check if file exists and get its size
        should_prepend_newline = os.path.exists(filename) and os.path.getsize(filename) > 0

        with open(filename, 'a') as file:  # Open file in append mode
            # If file exists and is not empty, prepend newline, otherwise just write the integer
            file.write(f"\n{integer}" if should_prepend_newline else f"{integer}")
            return True
    except IOError as e:
        logger.log_error("Utils", f"Error while writing to the file: {filename}. Error: {e}")
        return False


def overwrite_file(filename, integer):
    """Overwrite the file with a new integer."""
    try:
        with open(filename, 'w') as file:  # 'w' mode will also overwrite the file if it already exists
            file.write(str(integer))
            return True
    except IOError:
        logger.log_error("Utils", "Error while writing to the file : " + filename)
        return False


def read_from_file(filename):
    """Read an integer from a file."""
    try:
        with open(filename, 'r') as file:  # 'r' mode is for reading
            return int(file.read().strip())
    except IOError:
        logger.log_error("Utils", "Error while reading from the file : " + filename)
        return -1

from dronebuddylib.utils import DroneCommands


def create_system_drone_action_list() -> str:
    list_actions = [e.name for e in DroneCommands]
    action_string = ""
    for action in list_actions:
        action_string = action_string + action + "\n"

    return action_string


def create_custom_drone_action_list(custom_actions: list) -> str:
    action_string = ""
    for action in custom_actions:
        action_string = action_string + action + "\n"

    return action_string

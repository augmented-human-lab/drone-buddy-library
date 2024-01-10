from dronebuddylib.models.enums import AtomicEngineConfigurations


class EngineConfigurations:
    def __init__(self, configurations: dict):
        self.configurations = configurations

    def add_configuration(self, key: AtomicEngineConfigurations, value: str):
        self.configurations[key] = value

    def remove_configurations(self, key: AtomicEngineConfigurations) -> str:
        return self.configurations.pop(key)

    def get_configuration(self, key: AtomicEngineConfigurations) -> str:
        return self.configurations.get(key)

    def get_configurations(self) -> dict:
        return self.configurations

    def get_configurations_for_engine(self, class_name: str) -> dict:
        # return all the key pairs which contains a given string as a part of the key
        filtered_items = {}

        for key, value in self.configurations.items():
            if class_name in key.name:
                filtered_items[key] = value

        return filtered_items

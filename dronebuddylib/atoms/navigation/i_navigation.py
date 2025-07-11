from abc import abstractmethod

from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.i_dbl_function import IDBLFunction


class INavigation(IDBLFunction):

    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Constructor to initialize the navigation engine.

        Args:
            engine_configurations (EngineConfigurations): The configurations for the navigation engine.
        """
        self.engine_configurations = engine_configurations

    @abstractmethod
    def map_location(self) -> list:
        pass

    @abstractmethod
    def navigate(self) -> list:
        pass

    @abstractmethod
    def navigate_to_waypoint(self, location, destination_waypoint) -> list:
        pass

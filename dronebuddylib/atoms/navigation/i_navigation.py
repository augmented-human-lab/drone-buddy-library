from abc import abstractmethod
from typing import Union, TYPE_CHECKING

from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.i_dbl_function import IDBLFunction

if TYPE_CHECKING:
    from dronebuddylib.atoms.navigation.tello_waypoint_nav_utils.tello_waypoint_nav_coordinator import NavigationInstruction


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
    def navigate_to_waypoint(self, destination_waypoint: str, instruction: 'NavigationInstruction') -> list:
        """
        Navigate to a specific waypoint with strict NavigationInstruction enum enforcement.
        
        Args:
            destination_waypoint (str): The waypoint to navigate to.
            instruction (NavigationInstruction): Must be NavigationInstruction.CONTINUE or NavigationInstruction.HALT.
        
        Returns:
            list: Result of navigation operation.
        
        Raises:
            TypeError: If instruction is not a NavigationInstruction enum.
            ValueError: If instruction is not CONTINUE or HALT.
        """
        pass

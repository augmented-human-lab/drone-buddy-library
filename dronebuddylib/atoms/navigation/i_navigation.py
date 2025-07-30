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
        """
        Allows user to map the current location and returns a list of waypoints.

        Returns:
            list: A list of waypoints representing the mapped location.
        """
        pass

    @abstractmethod
    def navigate(self) -> list:
        """
        Provides navigation interface to the user to navigate between known waypoints
        
        Returns: 
         list: A list of navigated waypoints.
        """
        pass

    @abstractmethod
    def navigate_to_waypoint(self, destination_waypoint: str, instruction: 'NavigationInstruction') -> list:
        """
        Navigates to a specific waypoint with strict NavigationInstruction enum enforcement.

        Args:
            destination_waypoint (str): The waypoint to navigate to.
            instruction (NavigationInstruction): Must be NavigationInstruction.CONTINUE or NavigationInstruction.HALT.

        Returns:
            list: Result of the navigation operation, first element is a boolean indicating if the drone has landed or not (True if landed, False if still flying), second element is the current waypoint of the drone. Can be used to determine if the drone has successfully navigated to the destination waypoint.

        Raises:
            TypeError: If instruction is not a NavigationInstruction enum.
        """

        pass

    @abstractmethod
    def navigate_to(self, waypoints: list, final_instruction: 'NavigationInstruction') -> list: 
        """
        Navigates to a sequence of waypoints with strict NavigationInstruction enum enforcement.

        Args:
            waypoints (list): List of waypoints to navigate to.
            final_instruction (NavigationInstruction): Must be NavigationInstruction.CONTINUE or NavigationInstruction.HALT.

        Returns:
            list: Contains the list of waypoints the drone has navigated to. 
            
        Raises:
            TypeError: If final_instruction is not a NavigationInstruction enum.
        """
        pass

from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import NavigationAlgorithm
from dronebuddylib.utils.logger import Logger

logger = Logger()

class NavigationEngine:
    
    def __init__(self, algorithm: NavigationAlgorithm, config: EngineConfigurations):
        """
        Initializes the navigation engine with the specified algorithm and configuration.

        Args:
            algorithm (NavigationAlgorithm): The navigation algorithm to be used.
            config (EngineConfigurations): The configuration for the navigation engine.
        """
        if algorithm == NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT or algorithm == NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT.name:
            logger.log_info(self.get_class_name(), 'Preparing to initialize Tello Waypoint navigation engine.')
            from dronebuddylib.atoms.navigation.tello_navigation_impl import NavigationWaypointImpl
            self.navigation_engine = NavigationWaypointImpl(config)
        else:
            raise ValueError(f"Unsupported navigation algorithm: {algorithm}")

    def get_class_name(self) -> str:
        """
        Returns the class name.

        Returns:
            str: The class name.
        """
        return 'NAVIGATION_ENGINE'

    def map_location(self) -> list:
        """
        Retrieves the current map location.

        Returns:
            list: The current map location.
        """
        return self.navigation_engine.map_location()

    def navigate(self) -> list:
        """
        Initiates navigation.

        Returns:
            list: The result of the navigation operation.
        """
        return self.navigation_engine.navigate()

    def navigate_to_waypoint(self, destination_waypoint, instruction) -> list:
        """
        Navigates to a specific waypoint.

        Args:
            destination_waypoint: The waypoint to navigate to.
            instruction: The instruction for navigation.

        Returns:
            list: The result of the navigation to the specified waypoint.
        """
        return self.navigation_engine.navigate_to_waypoint(destination_waypoint, instruction)
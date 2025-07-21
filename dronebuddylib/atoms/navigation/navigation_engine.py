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
            logger.log_debug(self.get_class_name(), 'Tello Waypoint navigation engine initialized successfully.')
        else:
            logger.log_error(self.get_class_name(), f'Unsupported navigation algorithm: {algorithm}')
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
        logger.log_debug(self.get_class_name(), 'Starting map location operation.')
        result = self.navigation_engine.map_location()
        logger.log_debug(self.get_class_name(), f'Map location operation completed. Mapped {len(result)} waypoints.')
        return result

    def navigate(self) -> list:
        """
        Initiates navigation.

        Returns:
            list: The result of the navigation operation.
        """
        logger.log_info(self.get_class_name(), 'Starting navigation operation.')
        result = self.navigation_engine.navigate()
        logger.log_debug(self.get_class_name(), f'Navigation operation completed with {len(result)} results.')
        return result

    def navigate_to_waypoint(self, destination_waypoint, instruction) -> list:
        """
        Navigates to a specific waypoint with strict NavigationInstruction enum enforcement.

        Args:
            destination_waypoint (str): The waypoint to navigate to.
            instruction (NavigationInstruction): Must be NavigationInstruction.CONTINUE or NavigationInstruction.HALT.

        Returns:
            list: The result of the navigation to the specified waypoint.
            
        Raises:
            TypeError: If instruction is not a NavigationInstruction enum.
            ValueError: If instruction is not CONTINUE or HALT.
        """
        
        logger.log_info(self.get_class_name(), f'Starting navigation to waypoint: {destination_waypoint}')
        logger.log_debug(self.get_class_name(), f'Navigation instruction: {instruction}')
        
        result = self.navigation_engine.navigate_to_waypoint(destination_waypoint, instruction)
        
        logger.log_debug(self.get_class_name(), f'Navigate to waypoint operation completed with drone at current waypoint: {result[0]}.')
        return result
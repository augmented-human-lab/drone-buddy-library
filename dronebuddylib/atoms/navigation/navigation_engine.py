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
        Allows user to map the current location and returns a list of waypoints.

        Returns:
            list: A list of waypoints representing the mapped location.
        """
        logger.log_debug(self.get_class_name(), 'Starting map location operation.')
        result = self.navigation_engine.map_location()
        logger.log_debug(self.get_class_name(), f'Map location operation completed. Mapped {len(result)} waypoints.')
        return result

    def navigate(self) -> list:
        """
        Provides navigation interface to the user to navigate between known waypoints
        
        Returns: 
         list: A list of navigated waypoints.
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
            list: Result of the navigation operation, first element is a boolean indicating if the drone has landed or not (True if landed, False if still flying), second element is the current waypoint of the drone. Can be used to determine if the drone has successfully navigated to the destination waypoint.

        Raises:
            TypeError: If instruction is not a NavigationInstruction enum.
        """
        
        logger.log_info(self.get_class_name(), f'Starting navigation to waypoint: {destination_waypoint}')
        logger.log_debug(self.get_class_name(), f'Navigation instruction: {instruction}')
        
        result = self.navigation_engine.navigate_to_waypoint(destination_waypoint, instruction)
        
        logger.log_debug(self.get_class_name(), f'Navigate to waypoint operation completed with drone at current waypoint: {result[0]}.')
        return result
    
    def navigate_to(self, waypoints, final_instruction): 
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
        
        logger.log_info(self.get_class_name(), f'Starting navigation to waypoints: {waypoints}')
        logger.log_debug(self.get_class_name(), f'Final navigation instruction: {final_instruction}')
        
        result = self.navigation_engine.navigate_to(waypoints, final_instruction)
        
        logger.log_debug(self.get_class_name(), f'Navigate to waypoints operation completed with drone at current waypoint: {result[0]}.')
        return result
    
    def scan_surrounding(self) -> list:
        """
        Performs a surrounding scan operation using the Tello drone.

        Args:
            coordinator_instance (Coordinator): The coordinator instance managing the drone and waypoints.

        Returns:
            list: A list of images captured during the scan.
        """
        
        logger.log_info(self.get_class_name(), 'Starting surrounding scan operation.')

        result = self.navigation_engine.scan_surrounding()

        logger.log_debug(self.get_class_name(), f'Surrounding scan operation completed with {len(result)} images captured.')
        return result
    
    def get_drone_instance(self):
        """
        Returns the Tello drone instance.

        Returns:
            Optional[Tello]: The Tello drone instance if available, otherwise None.
        """
        
        logger.log_info(self.get_class_name(), 'Retrieving Tello drone instance.')
        
        drone = self.navigation_engine.get_drone_instance()

        logger.log_debug(self.get_class_name(), f'Drone instance retrieved: {drone is not None}.')
        
        return drone

    def takeoff(self) -> bool:
        """
        Initiates the takeoff sequence for the drone.

        Returns:
            bool: True if the takeoff was successful, False otherwise.
        """
        logger.log_info(self.get_class_name(), 'Starting takeoff operation.')

        result = self.navigation_engine.takeoff()

        logger.log_debug(self.get_class_name(), f'Takeoff operation completed with success: {result}.')
        return result

    def land(self) -> bool:
        """
        Initiates the landing sequence for the drone.

        Returns:
            bool: True if the landing was successful, False otherwise.
        """
        logger.log_info(self.get_class_name(), 'Starting landing operation.')

        result = self.navigation_engine.land()

        logger.log_debug(self.get_class_name(), f'Landing operation completed with success: {result}.')
        return result
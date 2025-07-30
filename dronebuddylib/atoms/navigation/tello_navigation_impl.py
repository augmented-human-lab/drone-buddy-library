import os
from dronebuddylib.atoms.navigation.i_navigation import INavigation
from dronebuddylib.models.enums import AtomicEngineConfigurations
from dronebuddylib.utils.logger import Logger
from dronebuddylib.utils.utils import config_validity_check
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.atoms.navigation.tello_waypoint_nav_utils.tello_waypoint_nav_coordinator import TelloWaypointNavCoordinator, NavigationInstruction

logger = Logger()

class NavigationWaypointImpl(INavigation):
    def map_location(self) -> list:
        """
        Allows user to map the current location and returns a list of waypoints.

        Returns:
            list: A list of waypoints representing the mapped location.
        """
        logger.log_info(self.get_class_name(), 'Starting location mapping operation.')
        logger.log_debug(self.get_class_name(), f'Using waypoint directory: {self.waypoint_dir}')
        
        coordinator = TelloWaypointNavCoordinator(self.waypoint_dir, self.vertical_factor,
                                                  self.mapping_movement_speed, self.mapping_rotation_speed, self.nav_speed, "mapping")
        result = coordinator.run()
        
        logger.log_info(self.get_class_name(), f'Location mapping session closed with {len(result)} waypoints.')
        return result

    def navigate(self) -> list:
        """
        Provides navigation interface to the user to navigate between known waypoints
        
        Returns: 
         list: A list of navigated waypoints.
        """
        logger.log_info(self.get_class_name(), 'Starting navigation between waypoints.')
        logger.log_debug(self.get_class_name(), f'Using waypoint directory: {self.waypoint_dir}')
        
        coordinator = TelloWaypointNavCoordinator(self.waypoint_dir, self.vertical_factor,
                                                  self.mapping_movement_speed, self.mapping_rotation_speed, self.nav_speed, "navigation", waypoint_file=self.waypoint_file)
        result = coordinator.run()
        
        logger.log_info(self.get_class_name(), f'Navigation session closed with drone travelled to {len(result)} waypoints.')
        return result

    def navigate_to_waypoint(self, destination_waypoint: str, instruction: NavigationInstruction) -> list:
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
        
        create_new = False
        coordinator_instance = TelloWaypointNavCoordinator._active_instance

        # Strict type enforcement
        if not isinstance(instruction, NavigationInstruction):
            error_msg = f"instruction must be a NavigationInstruction enum, got {type(instruction).__name__}: {instruction}"
            logger.log_error(self.get_class_name(), error_msg)
            if coordinator_instance is not None:
                coordinator_instance.is_goto_mode = False
                coordinator_instance.is_running = False
                coordinator_instance.cleanup()
            raise TypeError(error_msg)
        
        if coordinator_instance is None: 
            create_new = True

        coordinator = TelloWaypointNavCoordinator.get_instance(self.waypoint_dir, self.vertical_factor, self.mapping_movement_speed, self.mapping_rotation_speed, self.nav_speed, "goto", destination_waypoint, instruction, self.waypoint_file, create_new)
        result = coordinator.run()
        
        logger.log_info(self.get_class_name(), f'Navigation to waypoint session closed with drone at current waypoint: {result[1]}.')
        return result

    def navigate_to(self, waypoints: list, final_instruction: NavigationInstruction) -> list:
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
        
        coordinator_instance = TelloWaypointNavCoordinator._active_instance

        # Strict type enforcement
        if not isinstance(final_instruction, NavigationInstruction):
            error_msg = f"final_instruction must be a NavigationInstruction enum, got {type(final_instruction).__name__}: {final_instruction}"
            logger.log_error(self.get_class_name(), error_msg)
            if coordinator_instance is not None:
                coordinator_instance.is_goto_mode = False
                coordinator_instance.is_running = False
                coordinator_instance.cleanup()
            raise TypeError(error_msg)
        
        # Validate waypoints list
        if not waypoints:
            error_msg = "waypoints list cannot be empty"
            logger.log_error(self.get_class_name(), error_msg)
            raise ValueError(error_msg)

        accumulated_results = []

        for i, waypoint in enumerate(waypoints):
            is_last_waypoint = (i == len(waypoints) - 1)
            
            # Use CONTINUE for all waypoints except the last one
            if is_last_waypoint:
                current_instruction = final_instruction
                logger.log_debug(self.get_class_name(), f'Final waypoint {waypoint}: using {final_instruction}')
            else:
                current_instruction = NavigationInstruction.CONTINUE
                logger.log_debug(self.get_class_name(), f'Intermediate waypoint {waypoint}: using CONTINUE')
            
            # Navigate to current waypoint
            logger.log_info(self.get_class_name(), f'Navigating to waypoint {i+1}/{len(waypoints)}: {waypoint}')
            
            try:
                result = self.navigate_to_waypoint(waypoint, current_instruction)
                if result[0] and current_instruction == NavigationInstruction.CONTINUE: 
                    logger.log_error(self.get_class_name(), f"Navigation to waypoint {waypoint} failed, drone landed unexpectedly.")
                    break
                accumulated_results.extend([result[1]])
                logger.log_info(self.get_class_name(), f'Drone currently at waypoint {result[1]}')
            except Exception as e:
                logger.log_error(self.get_class_name(), f'Failed to reach waypoint {waypoint}: {e}')
                # You can choose to break here or continue to next waypoint
                break  # Stop navigation on first failure

        logger.log_info(self.get_class_name(), f'Navigation to waypoints session closed with drone at current waypoint: {accumulated_results[len(accumulated_results) - 1]}.')
        return accumulated_results

    def get_required_params(self) -> list:
        return []

    def get_optional_params(self) -> list:
        return [AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_DIR, AtomicEngineConfigurations.NAVIGATION_TELLO_VERTICAL_FACTOR,
                AtomicEngineConfigurations.NAVIGATION_TELLO_MAPPING_MOVEMENT_SPEED, AtomicEngineConfigurations.NAVIGATION_TELLO_MAPPING_ROTATION_SPEED,
                AtomicEngineConfigurations.NAVIGATION_TELLO_NAVIGATION_SPEED, AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_FILE]

    def get_class_name(self) -> str:
        """
        Returns the class name.

        Returns:
            str: The class name.
        """
        return 'NAVIGATION_TELLO_WAYPOINT'

    def get_algorithm_name(self) -> str:
        """
        Returns the algorithm name.

        Returns:
            str: The algorithm name.
        """
        return 'Tello Waypoint Navigation'

    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Initializes the tello navigation engine with the given configurations.

        Args:
            engine_configurations (EngineConfigurations): The engine configurations.
        """
        logger.log_info(self.get_class_name(), 'Initializing Tello navigation engine.')
        
        super().__init__(engine_configurations)
        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())
        
        configs = engine_configurations.get_configurations_for_engine(self.get_class_name())
        self.waypoint_dir = configs.get(AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_DIR)
        if self.waypoint_dir is None: 
            self.waypoint_dir = os.path.join(os.path.expanduser("~"), "dronebuddylib", "tellowaypoints")
        
        os.makedirs(self.waypoint_dir, exist_ok=True)
        logger.log_debug(self.get_class_name(), f"Waypoint directory set to: {self.waypoint_dir}")
        
        self.vertical_factor = configs.get(AtomicEngineConfigurations.NAVIGATION_TELLO_VERTICAL_FACTOR, 1.0)
        self.mapping_movement_speed = configs.get(AtomicEngineConfigurations.NAVIGATION_TELLO_MAPPING_MOVEMENT_SPEED, 38)
        self.mapping_rotation_speed = configs.get(AtomicEngineConfigurations.NAVIGATION_TELLO_MAPPING_ROTATION_SPEED, 70)
        self.nav_speed = configs.get(AtomicEngineConfigurations.NAVIGATION_TELLO_NAVIGATION_SPEED, 55)
        self.waypoint_file = configs.get(AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_FILE, None)
        
        logger.log_info(self.get_class_name(), 'Tello navigation engine initialized successfully.')
        logger.log_debug(self.get_class_name(), f'Configuration: vertical_factor={self.vertical_factor}, mapping_movement_speed={self.mapping_movement_speed}, mapping_rotation_speed={self.mapping_rotation_speed}, nav_speed={self.nav_speed}')


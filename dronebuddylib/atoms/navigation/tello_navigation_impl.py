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
                                                  self.mapping_movement_speed, self.mapping_rotation_speed, self.nav_speed, "navigation")
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
            list: Containing the drone's current waypoint
            
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

        coordinator = TelloWaypointNavCoordinator.get_instance(self.waypoint_dir, self.vertical_factor, self.mapping_movement_speed, self.mapping_rotation_speed, self.nav_speed, "goto", destination_waypoint, instruction, create_new)
        result = coordinator.run()
        
        logger.log_info(self.get_class_name(), f'Navigation to waypoint session closed with drone at current waypoint: {result[0]}.')
        return result

    def get_required_params(self) -> list:
        return []

    def get_optional_params(self) -> list:
        return [AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_DIR, AtomicEngineConfigurations.NAVIGATION_TELLO_VERTICAL_FACTOR,
                AtomicEngineConfigurations.NAVIGATION_TELLO_MAPPING_MOVEMENT_SPEED, AtomicEngineConfigurations.NAVIGATION_TELLO_MAPPING_ROTATION_SPEED,
                AtomicEngineConfigurations.NAVIGATION_TELLO_NAVIGATION_SPEED]

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
        
        logger.log_info(self.get_class_name(), 'Tello navigation engine initialized successfully.')
        logger.log_debug(self.get_class_name(), f'Configuration: vertical_factor={self.vertical_factor}, mapping_movement_speed={self.mapping_movement_speed}, mapping_rotation_speed={self.mapping_rotation_speed}, nav_speed={self.nav_speed}')


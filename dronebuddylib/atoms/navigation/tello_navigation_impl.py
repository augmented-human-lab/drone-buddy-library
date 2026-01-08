import os
import time
from typing import Optional

from djitellopy import Tello

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
        
        # Create coordinator instance specifically configured for mapping mode
        coordinator = TelloWaypointNavCoordinator(
            self.waypoint_dir,              # Directory where waypoint files will be saved
            self.vertical_factor,           # Vertical movement scaling factor for altitude control
            self.mapping_movement_speed,    # Speed for manual mapping movements (cm/s)
            self.mapping_rotation_speed,    # Rotation speed for directional changes during mapping
            self.nav_speed,                 # Speed for potential navigation operations
            "mapping"                       # Set operational mode to mapping for manual control
        )
        result = coordinator.run()  # Execute mapping mode with real-time manual controls
        
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
        
        # Create coordinator instance configured for interactive navigation mode
        coordinator = TelloWaypointNavCoordinator(
            self.waypoint_dir,              # Directory containing waypoint files for navigation
            self.vertical_factor,           # Vertical movement scaling factor for altitude adjustments
            self.mapping_movement_speed,    # Speed configuration (inherited from mapping settings)
            self.mapping_rotation_speed,    # Rotation speed configuration for navigation
            self.nav_speed,                 # Primary navigation speed for autonomous movement
            "navigation",                   # Set operational mode to navigation for interactive selection
            waypoint_file=self.waypoint_file  # Optional specific waypoint file to use
        )
        result = coordinator.run()  # Execute navigation mode with interactive waypoint selection
        
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
        
        create_new = False  # Flag to determine if new coordinator instance needs to be created
        coordinator_instance = TelloWaypointNavCoordinator._active_instance  # Get current active singleton instance

        # Strict type enforcement for navigation instruction parameter
        if not isinstance(instruction, NavigationInstruction):
            error_msg = f"instruction must be a NavigationInstruction enum, got {type(instruction).__name__}: {instruction}"
            logger.log_error(self.get_class_name(), error_msg)
            # Cleanup existing coordinator if present due to parameter error
            if coordinator_instance is not None:
                coordinator_instance.is_goto_mode = False  # Disable goto mode
                coordinator_instance.is_running = False    # Stop coordinator operations
                coordinator_instance.cleanup()             # Perform resource cleanup
            raise TypeError(error_msg)
        
        # Check if coordinator instance exists - if not, create new one
        if coordinator_instance is None: 
            create_new = True

        # Get or create coordinator instance with goto mode configuration
        coordinator = TelloWaypointNavCoordinator.get_instance(
            self.waypoint_dir,              # Directory for waypoint file storage
            self.vertical_factor,           # Vertical movement scaling factor
            self.mapping_movement_speed,    # Speed for mapping operations
            self.mapping_rotation_speed,    # Rotation speed for mapping
            self.nav_speed,                 # Navigation movement speed
            "goto",                         # Set mode to goto for direct waypoint navigation
            destination_waypoint,           # Target waypoint identifier
            instruction,                    # Post-arrival behavior instruction
            self.waypoint_file,             # Specific waypoint file to use
            create_new                      # Whether to force new instance creation
        )
        result = coordinator.run()  # Execute goto mode navigation operation
        
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
        
        coordinator_instance = TelloWaypointNavCoordinator._active_instance  # Get current singleton instance

        # Strict type enforcement for final instruction parameter
        if not isinstance(final_instruction, NavigationInstruction):
            error_msg = f"final_instruction must be a NavigationInstruction enum, got {type(final_instruction).__name__}: {final_instruction}"
            logger.log_error(self.get_class_name(), error_msg)
            # Cleanup coordinator if present due to parameter validation failure
            if coordinator_instance is not None:
                coordinator_instance.is_goto_mode = False  # Disable goto mode
                coordinator_instance.is_running = False    # Stop all coordinator operations
                coordinator_instance.cleanup()             # Clean up resources
            raise TypeError(error_msg)
        
        # Validate waypoints list is not empty, return empty list and exit function if it is
        if not waypoints:
            error_msg = "waypoints list cannot be empty"
            logger.log_error(self.get_class_name(), error_msg)
            return []

        accumulated_results = []  # Store waypoints that drone successfully navigated to

        # Iterate through each waypoint in the sequence
        for i, waypoint in enumerate(waypoints):
            is_last_waypoint = (i == len(waypoints) - 1)  # Check if this is the final waypoint
            
            # Determine instruction for current waypoint based on position in sequence
            if is_last_waypoint:
                current_instruction = final_instruction  # Use final instruction for last waypoint
                logger.log_debug(self.get_class_name(), f'Final waypoint {waypoint}: using {final_instruction}')
            else:
                current_instruction = NavigationInstruction.CONTINUE  # Keep flying for intermediate waypoints
                logger.log_debug(self.get_class_name(), f'Intermediate waypoint {waypoint}: using CONTINUE')
            
            # Navigate to current waypoint using single waypoint navigation method
            logger.log_info(self.get_class_name(), f'Navigating to waypoint {i+1}/{len(waypoints)}: {waypoint}')
            
            try:
                # Execute navigation to current waypoint with appropriate instruction
                result = self.navigate_to_waypoint(waypoint, current_instruction)
                # Check if drone landed unexpectedly when it should continue flying
                if result[0] and current_instruction == NavigationInstruction.CONTINUE: 
                    logger.log_error(self.get_class_name(), f"Navigation to waypoint {waypoint} failed, drone landed unexpectedly.")
                    break  # Stop sequence navigation on unexpected landing
                accumulated_results.extend([result[1]])  # Add reached waypoint to results
                logger.log_info(self.get_class_name(), f'Drone currently at waypoint {result[1]}')
            except Exception as e:
                logger.log_error(self.get_class_name(), f'Failed to reach waypoint {waypoint}: {e}')
                # Stop navigation sequence on first failure to ensure safety
                break  # Stop navigation on first failure

        logger.log_info(self.get_class_name(), f'Navigation to waypoints session closed with drone at current waypoint: {accumulated_results[len(accumulated_results) - 1]}.')
        return accumulated_results
    
    def scan_surrounding(self) -> list:
        """
        Takes pictures of the surrounding of the drone while doing a 360 degree rotation.
        
        Returns:
            list: A list of images captured during the scan.
        """
        coordinator_instance = TelloWaypointNavCoordinator._active_instance  # Get current singleton instance
        if coordinator_instance is None: 
            logger.log_error(self.get_class_name(), 'No active drone or drone is not flying to perform surrounding scan.')
            return []
        
        logger.log_info(self.get_class_name(), 'Starting surrounding scan operation.')
        current_waypoint_file = coordinator_instance.waypoint_file  # Get current waypoint file from coordinator
        current_waypoint = coordinator_instance.current_waypoint  # Get current waypoint from coordinator
        coordinator_instance._pause_battery_monitoring()  # Pause battery monitoring during scan to prevent conflicts

        time.sleep(0.25)  # Small delay to ensure battery monitoring is paused before scan

        # Check if emergency shutdown is triggered before proceeding with scan
        if coordinator_instance._emergency_shutdown: 
            Logger.log_error(self.get_class_name(), 'Emergency shutdown detected - stopping surrounding scan.')
            return []
        
        from dronebuddylib.atoms.navigation.tello_waypoint_nav_utils.tello_nav_extra import TelloNavExtra
        tello_manouver = TelloNavExtra(coordinator_instance.tello, self.image_dir)  # Use existing drone instance from coordinator
        
        # Perform surrounding scan operation using TelloNavExtra utility
        result = tello_manouver.scan(current_waypoint_file, current_waypoint)
        logger.log_info(self.get_class_name(), f'Surrounding scan operation completed with {len(result)} images captured.')

        coordinator_instance._resume_battery_monitoring()  # Resume battery monitoring after scan
        return result
    
    def get_drone_instance(self) -> Optional[Tello]:
        """
        Returns the Tello drone instance.

        Returns:
            Optional[Tello]: The Tello drone instance if available, otherwise None.
        """
        # Get the active coordinator instance to access the Tello drone
        coordinator_instance = TelloWaypointNavCoordinator._active_instance

        # If no active coordinator instance, return None
        if coordinator_instance is None:
            logger.log_warning(self.get_class_name(), 'No active drone instance available.')
            return None
        
        # Return the Tello drone instance from the coordinator
        return coordinator_instance.tello

    def takeoff(self) -> bool:
        """
        Initiates the takeoff sequence for the drone.

        Returns:
            bool: True if the takeoff was successful, False otherwise.
        """
        coordinator_instance = TelloWaypointNavCoordinator._active_instance  # Get current singleton instance

        if coordinator_instance is not None: 
            logger.log_info(self.get_class_name(), 'Drone is already flying. ')
            return False  # Drone is already flying, cannot take off again

        # Drone must always be placed at starting waypoint: WP_001 for takeoff
        # Calling navigate_to_waypoint at WP_001 and NavigationInstruction.CONTINUE when the drone was uninitialized will cause the drone to simply takeoff and hover at its current position which is assumed to be WP_001
        result = self.navigate_to_waypoint("WP_001", NavigationInstruction.CONTINUE)

        if result[0]:
            return False # Drone landed instead of having completed the takeoff operation 
        else: 
            return True  # Takeoff operation completed successfully: drone is still hovering at WP_001
    
    def land(self) -> bool: 
        """
        Initiates the landing sequence for the drone.

        Returns:
            bool: True if the landing was successful, False otherwise.
        """
        coordinator_instance = TelloWaypointNavCoordinator._active_instance  # Get current singleton instance

        if coordinator_instance is None: 
            logger.log_warning(self.get_class_name(), 'No active drone instance available for landing. Drone already landed.')
            return False  # Drone already landed, cannot land again

        # Call navigate_to_waypoint at current position and NavigationInstruction.LAND to initiate landing at current position/waypoint
        result = self.navigate_to_waypoint(coordinator_instance.current_waypoint, NavigationInstruction.HALT)

        if result[0]:
            return True  # Drone landed successfully
        else: 
            return False # Drone still not landed

    def get_required_params(self) -> list:
        """
        Returns a list of required parameters for this navigation engine.

        Returns:
            list: A list of required parameters.
        """
        return []

    def get_optional_params(self) -> list:
        """
        Returns a list of optional parameters for this navigation engine.

        Returns:
            list: A list of optional parameters.
        """
        return [AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_DIR, AtomicEngineConfigurations.NAVIGATION_TELLO_VERTICAL_FACTOR,
                AtomicEngineConfigurations.NAVIGATION_TELLO_MAPPING_MOVEMENT_SPEED, AtomicEngineConfigurations.NAVIGATION_TELLO_MAPPING_ROTATION_SPEED,
                AtomicEngineConfigurations.NAVIGATION_TELLO_NAVIGATION_SPEED, AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_FILE, AtomicEngineConfigurations.NAVIGATION_TELLO_IMAGE_DIR]

    def get_class_name(self) -> str:
        """
        Returns the class name.

        Returns:
            str: The class name.
        """
        # Return unique identifier for this navigation implementation
        return 'NAVIGATION_TELLO_WAYPOINT'

    def get_algorithm_name(self) -> str:
        """
        Returns the algorithm name.

        Returns:
            str: The algorithm name.
        """
        # Return human-readable algorithm name for logging and identification
        return 'Tello Waypoint Navigation'

    def __init__(self, engine_configurations: EngineConfigurations):
        """
        Initializes the tello navigation engine with the given configurations.

        Args:
            engine_configurations (EngineConfigurations): The engine configurations.
        """
        logger.log_info(self.get_class_name(), 'Initializing Tello navigation engine.')
        
        super().__init__(engine_configurations)  # Initialize parent INavigation interface
        # Validate required configuration parameters are present
        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())
        
        # Extract configuration parameters for this navigation engine
        configs = engine_configurations.get_configurations_for_engine(self.get_class_name())
        
        # Configure waypoint directory with default fallback to user home directory
        self.waypoint_dir = configs.get(AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_DIR)
        if self.waypoint_dir is None: 
            # Use default directory in user home folder if not specified
            self.waypoint_dir = os.path.join(os.path.expanduser("~"), "dronebuddylib", "tellowaypoints")
        
        # Ensure waypoint directory exists, create if necessary
        os.makedirs(self.waypoint_dir, exist_ok=True)
        logger.log_debug(self.get_class_name(), f"Waypoint directory set to: {self.waypoint_dir}")
        
        # Configure movement and navigation parameters with default values
        self.vertical_factor = configs.get(AtomicEngineConfigurations.NAVIGATION_TELLO_VERTICAL_FACTOR, 1.0)  # Vertical movement scaling
        self.mapping_movement_speed = configs.get(AtomicEngineConfigurations.NAVIGATION_TELLO_MAPPING_MOVEMENT_SPEED, 38)  # Mapping speed (cm/s)
        self.mapping_rotation_speed = configs.get(AtomicEngineConfigurations.NAVIGATION_TELLO_MAPPING_ROTATION_SPEED, 70)  # Rotation speed (deg/s)
        self.nav_speed = configs.get(AtomicEngineConfigurations.NAVIGATION_TELLO_NAVIGATION_SPEED, 55)  # Navigation speed (cm/s)
        self.waypoint_file = configs.get(AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_FILE, None)  # Optional specific waypoint file
        self.image_dir = configs.get(AtomicEngineConfigurations.NAVIGATION_TELLO_IMAGE_DIR, None)  # Directory for captured images
        
        logger.log_info(self.get_class_name(), 'Tello navigation engine initialized successfully.')
        logger.log_debug(self.get_class_name(), f'Configuration: vertical_factor={self.vertical_factor}, mapping_movement_speed={self.mapping_movement_speed}, mapping_rotation_speed={self.mapping_rotation_speed}, nav_speed={self.nav_speed}')


"""
Main coordinator for DJI Tello drone waypoint navigation system.

This module serves as the central orchestrator for all drone navigation operations, managing
three distinct operational modes and providing comprehensive safety features. It handles
cross-platform compatibility, drone lifecycle management, and emergency safety protocols.

Operational Modes:
- MAPPING: Manual flight control to create waypoint maps through real-time recording
- NAVIGATION: Interactive waypoint selection and autonomous navigation between recorded points
- GOTO: Direct navigation to specific waypoints with instruction-based control flow

Key Features:
- Cross-platform support (Windows/Linux) with platform-specific controllers
- Background battery monitoring with automatic emergency landing
- Emergency shutdown system for critical safety situations
- Singleton pattern for instance management and state persistence
- Thread-safe operations with battery monitoring pause/resume capabilities

Architecture:
- Factory pattern for instance creation and reuse
- State machine for mode transitions and lifecycle management
- Event-driven battery monitoring with configurable thresholds
- Resource cleanup and graceful shutdown handling
"""

from djitellopy import Tello
import time
import threading
import os
import glob
import sys
import platform
from enum import Enum

# Platform-specific imports for cross-platform compatibility
if platform.system() == 'Linux':
    from .realtime_drone_control import RealTimeDroneController
    from .navigation_interface import NavigationInterface
if platform.system() == 'Windows':
    from .realtime_drone_control_windows import RealTimeDroneControllerWindows
    from .navigation_interface_windows import NavigationInterfaceWindows
    
from dronebuddylib.utils.logger import Logger

logger = Logger()

class NavigationInstruction(Enum):
    """
    Navigation instruction enumeration for goto mode operation control.
    
    Defines the behavior after reaching a target waypoint during goto mode navigation.
    Used by the navigation engine to determine whether to land or continue operation.
    """
    CONTINUE = "continue"  # Keep drone flying and maintain session after reaching waypoint
    HALT = "halt"         # Land drone and terminate session after reaching waypoint

class TelloWaypointNavCoordinator: 
    """
    Central coordinator for DJI Tello drone waypoint navigation system.
    
    This class orchestrates all aspects of drone navigation operations, from initial mapping
    through autonomous navigation execution. It implements a comprehensive safety framework
    with battery monitoring, emergency shutdown, and graceful resource management.
    
    The coordinator operates in three distinct modes:
    - MAPPING: Real-time manual control for waypoint creation and map building
    - NAVIGATION: Interactive waypoint navigation with user selection interface
    - GOTO: Direct autonomous navigation to specific waypoints with instruction control
    
    Safety Features:
    - Background battery monitoring with automatic emergency landing (< 10% battery)
    - Emergency shutdown system for critical situations
    - Graceful cleanup and resource deallocation on exit
    - Thread-safe battery monitoring with pause/resume capabilities during navigation
    
    Architecture:
    - Singleton pattern with factory method for instance management
    - Cross-platform compatibility with OS-specific controller selection
    - State machine for mode transitions and operational status tracking
    - Event-driven monitoring system with configurable thresholds
    """

    # Class-level variables for singleton pattern and safety monitoring
    _active_instance = None               # Singleton instance tracker
    _battery_thread = None               # Background battery monitoring thread
    _battery_thread_running = False      # Control flag for battery monitoring loop
    _battery_monitoring_paused = False   # Pause flag for navigation operations
    _emergency_shutdown = False          # Emergency shutdown trigger for critical situations

    @classmethod
    def get_instance(cls, waypoint_dir: str, vertical_factor: float, movement_speed: int, rotation_speed: int, navigation_speed: int, mode: str, waypoint_dest: str = None, instruction: NavigationInstruction = None, waypoint_file: str = None, create_new: bool = False):
        """
        Factory method for singleton instance management with parameter-driven configuration.
        
        Manages coordinator lifecycle with comprehensive parameter validation and mode-specific
        configuration. Supports forced instance recreation or returns existing singleton with
        validation of compatibility between current and requested parameters.
        
        Args:
            waypoint_dir (str): Base directory for waypoint file storage and management
            vertical_factor (float): Vertical movement scaling factor for altitude adjustments
            movement_speed (int): Base movement speed for mapping operations (cm/s)
            rotation_speed (int): Rotation speed for directional changes (degrees/s)
            navigation_speed (int): Movement speed during autonomous navigation (cm/s)
            mode (str): Operational mode - 'mapping', 'navigation', or 'goto'
            waypoint_dest (str, optional): Target waypoint identifier for goto mode
            instruction (NavigationInstruction, optional): Navigation control instruction for goto mode
            waypoint_file (str, optional): Specific waypoint file for navigation operations
            create_new (bool): Force creation of new instance, replacing existing singleton
        
        Returns:
            TelloWaypointNavCoordinator: Configured singleton coordinator instance
            
        Raises:
            ValueError: For invalid mode specifications or parameter combinations
            
        Thread Safety:
            Safe for concurrent access with singleton pattern protection
        """
        if create_new: 
            instance = cls(waypoint_dir, vertical_factor, movement_speed, rotation_speed, navigation_speed, mode, waypoint_dest, instruction, waypoint_file)
            cls._active_instance = instance
            return instance
        else: 
            # Update parameters of existing instance
            instance = cls._active_instance
            instance.waypoint_dest = waypoint_dest
            instance.instruction = instruction
            return instance 

    def __init__(self, waypoint_dir: str, vertical_factor: float, movement_speed: int, rotation_speed: int, navigation_speed: int, mode: str, waypoint_dest: str = None, instruction: NavigationInstruction = None, waypoint_file: str = None):
        """
        Initialize coordinator with operational parameters and mode-specific configuration.
        
        Sets up the complete navigation environment including drone connection parameters,
        movement configurations, and operational mode preparation. Validates parameter
        compatibility and initializes cross-platform controller selection.
        
        Args:
            waypoint_dir (str): Base directory for waypoint file storage and management
            vertical_factor (float): Vertical movement scaling factor for altitude control
            movement_speed (int): Base movement speed for mapping operations (cm/s)
            rotation_speed (int): Rotation speed for directional adjustments (degrees/s)
            navigation_speed (int): Movement speed during autonomous navigation (cm/s)
            mode (str): Operational mode - 'mapping', 'navigation', or 'goto'
            waypoint_dest (str, optional): Target waypoint for goto mode operations
            instruction (NavigationInstruction, optional): Control instruction for goto navigation
            waypoint_file (str, optional): Specific waypoint file for navigation operations
            
        Raises:
            ValueError: For invalid operational mode or incompatible parameter combinations
            OSError: For waypoint directory access issues or platform compatibility problems
            
        Note:
            Drone connection is deferred until explicit connect_drone() call for resource management
        """
        logger.log_info('TelloWaypointNavCoordinator', f'Initializing coordinator in {mode} mode.')
        
        # Initialize parameters
        self.waypoint_dir = waypoint_dir
        self.vertical_factor = vertical_factor
        self.movement_speed = movement_speed
        self.rotation_speed = rotation_speed
        self.navigation_speed = navigation_speed
        self.mode = mode
        self.waypoint_dest = waypoint_dest
        self.instruction = instruction
        self.waypoint_file = waypoint_file

        # Initialize Tello drone
        logger.log_debug('TelloWaypointNavCoordinator', 'Initializing Tello drone.')
        self.tello = Tello()

        # Initialize application state
        self.is_connected = False
        self.is_flying = False
        self.is_mapping_mode = False
        self.is_navigation_mode = False
        self.is_goto_mode = False
        self.is_running = False
        self.current_waypoint = "WP_001"

        # Always reset the class-level variables when new instance is created
        TelloWaypointNavCoordinator._emergency_shutdown = False
        TelloWaypointNavCoordinator._battery_thread_running = False
        TelloWaypointNavCoordinator._battery_monitoring_paused = False
        TelloWaypointNavCoordinator._active_instance = None
        TelloWaypointNavCoordinator._battery_thread = None
        
        logger.log_debug('TelloWaypointNavCoordinator', f'Coordinator initialized with params: waypoint_dir={waypoint_dir}, mode={mode}, vertical_factor={vertical_factor}')
    
    def _start_battery_monitoring(self):
        """
        Initialize and start background battery monitoring thread for continuous safety oversight.
        
        Creates a daemon thread that continuously monitors drone battery level during operations.
        The monitoring system automatically triggers emergency landing procedures when battery
        drops below critical thresholds (10%) to prevent drone loss or damage.
        
        Thread Safety:
            Uses class-level flags to prevent multiple monitoring threads and ensure
            single-threaded battery monitoring across all coordinator instances.
            
        Lifecycle:
            - Daemon thread automatically terminates when main process exits
            - Can be paused during navigation operations to prevent interference
            - Includes graceful shutdown with timeout-based join for cleanup
        """
        if not TelloWaypointNavCoordinator._battery_thread_running:
            TelloWaypointNavCoordinator._battery_thread_running = True
            TelloWaypointNavCoordinator._battery_thread = threading.Thread(target=self._battery_monitor_loop, daemon=True)
            TelloWaypointNavCoordinator._battery_thread.start()
            logger.log_info('TelloWaypointNavCoordinator', 'Battery monitoring thread started.')
    
    def _stop_battery_monitoring(self):
        """
        Stop background battery monitoring thread with graceful cleanup.
        
        Safely terminates the battery monitoring thread using cooperative shutdown
        mechanisms. Includes timeout-based thread joining to prevent indefinite
        blocking during application shutdown.
        
        Cleanup Process:
            1. Set thread termination flag
            2. Wait for thread completion with 2-second timeout
            3. Log completion status for debugging
            
        Thread Safety:
            Safe to call multiple times - includes existence checks before operations
        """
        if TelloWaypointNavCoordinator._battery_thread_running:
            TelloWaypointNavCoordinator._battery_thread_running = False
            if TelloWaypointNavCoordinator._battery_thread and TelloWaypointNavCoordinator._battery_thread.is_alive():
                TelloWaypointNavCoordinator._battery_thread.join(timeout=2)
            logger.log_info('TelloWaypointNavCoordinator', 'Battery monitoring thread stopped.')
    
    @classmethod
    def _pause_battery_monitoring(cls):
        """
        Temporarily pause battery monitoring during critical navigation operations.
        
        Prevents battery monitoring interference during time-sensitive navigation
        sequences where telemetry polling might disrupt movement precision or
        introduce communication conflicts with the drone.
        
        Usage:
            Called automatically during waypoint navigation sequences to ensure
            smooth operation without battery monitoring interruptions.
            
        Thread Safety:
            Class-level method safe for concurrent access across all coordinator instances
        """
        cls._battery_monitoring_paused = True
        logger.log_debug('TelloWaypointNavCoordinator', 'Battery monitoring paused.')
    
    @classmethod
    def _resume_battery_monitoring(cls):
        """
        Resume battery monitoring after completion of navigation operations.
        
        Re-enables continuous battery monitoring following navigation sequences
        or other operations that required temporary monitoring suspension.
        Ensures safety oversight is restored promptly after critical operations.
        
        Usage:
            Called automatically after waypoint navigation completion to restore
            full safety monitoring capabilities.
            
        Thread Safety:
            Class-level method safe for concurrent access and state restoration
        """
        cls._battery_monitoring_paused = False
        logger.log_debug('TelloWaypointNavCoordinator', 'Battery monitoring resumed.')
    
    def _battery_monitor_loop(self):
        """
        Continuous battery monitoring loop for emergency safety management.
        
        Runs in background daemon thread to provide real-time battery oversight during
        all flight operations. Implements multi-tier warning system with automatic
        emergency procedures when battery reaches critical levels.
        
        Monitoring Logic:
        - 5-second polling interval for battery level checks
        - Respects pause flag during navigation operations
        - Warning threshold at 20% battery remaining
        - Critical emergency action at 10% battery remaining
        
        Emergency Response:
        - Immediate stop of all drone movements
        - Automatic emergency landing sequence
        - Graceful resource cleanup and disconnection
        - Program termination to prevent drone loss
        
        Thread Safety:
        - Uses class-level flags for coordinated shutdown
        - Exception handling prevents thread crashes
        - Timeout-based commands prevent hanging operations
        """
        while TelloWaypointNavCoordinator._battery_thread_running:
            try:
                # Check if battery monitoring is paused
                if TelloWaypointNavCoordinator._battery_monitoring_paused:
                    time.sleep(1)  # Short sleep when paused
                    continue
                    
                # Get the active instance to access the drone
                if TelloWaypointNavCoordinator._active_instance is None:
                    break
                
                instance = TelloWaypointNavCoordinator._active_instance
                if not instance.is_flying:
                    break
                
                battery_str = instance.tello.send_command_with_return("battery?", timeout=3)
                battery = int(battery_str)
                
                if battery < 20:
                    logger.log_warning('TelloWaypointNavCoordinator', f'Low battery detected: {battery}%')
                    if battery < 10:
                        logger.log_error('TelloWaypointNavCoordinator', f'CRITICAL: Battery too low ({battery}%), initiating emergency landing.')
                        # Set emergency shutdown flag for graceful shutdown path
                        TelloWaypointNavCoordinator._emergency_shutdown = True
                        # Force stop all operations
                        instance.is_running = False
                        instance.is_goto_mode = False
                        
                        # Stop any ongoing movement and land
                        try:
                            instance.tello.send_rc_control(0, 0, 0, 0)  # Stop immediately
                            time.sleep(0.5)  # Brief pause to ensure stop command is processed

                            if instance.is_flying:
                                logger.log_info('TelloWaypointNavCoordinator', 'Landing drone due to critical battery level.')
                                instance.land()
                        except Exception as e:
                            logger.log_error('TelloWaypointNavCoordinator', f'Error sending stop command: {e}')
                        
                        if instance.is_connected:
                            try:
                                logger.log_info('TelloWaypointNavCoordinator', 'Disconnecting from drone.')
                                instance.tello.end()
                                instance.is_connected = False
                            except Exception as e:
                                logger.log_error('TelloWaypointNavCoordinator', f'Error during disconnection: {e}')
                        
                        # Clear active instance
                        if TelloWaypointNavCoordinator._active_instance is not None:
                            TelloWaypointNavCoordinator._active_instance = None
                        # Stop battery monitoring
                        TelloWaypointNavCoordinator._battery_thread_running = False
                        logger.log_error('TelloWaypointNavCoordinator', 'EMERGENCY SHUTDOWN: Program terminating due to critical battery level')
                        # Force exit the entire program
                        sys.exit(1)
                        break
                
                # Sleep for 5 seconds before next check
                time.sleep(5)
                
            except Exception as e:
                logger.log_warning('TelloWaypointNavCoordinator', f'Battery check failed: {e}')
                time.sleep(5)
                continue
    
    def run(self):
        """
        Execute the main application workflow based on configured operational mode.
        
        Orchestrates the complete drone navigation lifecycle from initialization through
        cleanup, with mode-specific execution paths and comprehensive error handling.
        Ensures proper resource management and safe shutdown regardless of operation outcome.
        
        Operational Flow:
        1. Mode-specific execution (mapping/navigation/goto)
        2. Exception handling for user interruption and system errors
        3. Cleanup and landing operations in finally block
        4. Battery monitoring shutdown and instance cleanup
        
        Returns:
            list: Mode-specific execution summary:
                  - Mapping: List of created waypoints
                  - Navigation: Navigation history and visited waypoints
                  - Goto: Landing status and reached waypoint information
                  
        Error Handling:
        - KeyboardInterrupt: Graceful user-initiated shutdown
        - General exceptions: Error logging with safe cleanup
        - Finally block: Guaranteed resource deallocation
        """
        logger.log_info('TelloWaypointNavCoordinator', f'Starting navigation run in {self.mode} mode.')
        
        summary = []
        land = True  # Default to landing at the end
        try:
            if self.mode == "mapping":
                summary = self._run_mapping_mode()
            elif self.mode == "navigation":
                summary = self._run_navigation_mode()
            elif self.mode == "goto":
                summary = self.run_goto_mode()
                land = summary[0]  # Get land flag from goto mode
            
        except KeyboardInterrupt:
            logger.log_warning('TelloWaypointNavCoordinator', 'Application interrupted by user.')
            land = True  # Ensure we land on exit
        except Exception as e:
            logger.log_error('TelloWaypointNavCoordinator', f'Application error: {e}')
            land = True  # Ensure we land on error
        finally:
            if land: 
                self.is_running = False
                self.is_mapping_mode = False
                self.is_navigation_mode = False
                self._stop_battery_monitoring()  # Safe cleanup - only stops if running
                TelloWaypointNavCoordinator._active_instance = None
                self.cleanup()

            return summary

    def _run_mapping_mode(self) -> list:
        """
        Execute mapping mode for waypoint creation through manual drone control.
        
        Provides real-time manual control interface for drone operation, allowing users
        to fly the drone and create waypoint maps through keyboard controls. Automatically
        records movement sequences and waypoint locations for later navigation use.
        
        Workflow:
        1. Validate drone connection and flight status
        2. Display control instructions to user
        3. Establish drone connection and perform takeoff
        4. Initialize platform-specific real-time controller
        5. Execute manual control session with waypoint recording
        6. Handle cleanup and return waypoint creation summary
        
        Platform Support:
        - Windows: Enhanced controller with video streaming capabilities
        - Linux: Standard controller with keyboard input handling
        
        Returns:
            list: Summary of waypoints created during mapping session
            
        Safety Features:
        - Pre-flight connection and status validation
        - Automatic takeoff failure handling
        - Exception handling during controller execution
        - Guaranteed cleanup in finally block
        """
        logger.log_info('TelloWaypointNavCoordinator', 'MAPPING MODE ACTIVATED')
        print("You will create waypoints by manually controlling the drone.")

        if self.is_connected or self.is_flying:
            logger.log_warning('TelloWaypointNavCoordinator', 'Drone is already connected or flying. Please land it first.')
            return []

        self.display_controls()

        if not self.connect_drone():
            logger.log_error('TelloWaypointNavCoordinator', 'Failed to connect to drone. Exiting...')
            return []
        
        if not self.takeoff():
            logger.log_error('TelloWaypointNavCoordinator', 'Failed to take off. Exiting...')
            return []
        
        # Initialize drone controller based on OS
        current_os = platform.system()
        if current_os == 'Windows':
            logger.log_info('TelloWaypointNavCoordinator', 'Detected Windows OS - using Windows controller with video streaming')
            self.drone_controller = RealTimeDroneControllerWindows(self.waypoint_dir, self.movement_speed, self.rotation_speed)
        else:
            logger.log_info('TelloWaypointNavCoordinator', f'Detected {current_os} OS - using Linux controller')
            self.drone_controller = RealTimeDroneController(self.waypoint_dir, self.movement_speed, self.rotation_speed)
        
        self.is_mapping_mode = True
        self.is_running = True
    
        logger.log_info('TelloWaypointNavCoordinator', 'Use keyboard controls to move drone and create waypoints.')
        
        # Start user interface
        try:
            summary = self.drone_controller.run(drone_instance=self.tello)
        except Exception as e:
            logger.log_error('TelloWaypointNavCoordinator', f'Error during execution: {e}')
            summary = []
        finally:
            self.is_mapping_mode = False
            self.is_running = False

            return summary  # Return summary of waypoints created
    
    def _run_navigation_mode(self) -> list:
        """
        Execute interactive navigation mode for waypoint-based autonomous flight.
        
        Provides user interface for selecting waypoints from existing maps and
        executing autonomous navigation between selected points. Supports both
        specific waypoint file selection and automatic latest file discovery.
        
        Workflow:
        1. Validate drone connection and flight status
        2. Establish drone connection and perform takeoff
        3. Initialize platform-specific navigation interface
        4. Execute interactive waypoint selection and navigation
        5. Handle cleanup and return navigation history
        
        Platform Support:
        - Windows: Enhanced interface with platform-specific optimizations
        - Linux: Standard interface with cross-platform compatibility
        
        Returns:
            list: Navigation history including visited waypoints and travel paths
            
        Features:
        - Interactive waypoint selection from available maps
        - Autonomous navigation execution with safety monitoring
        - Real-time navigation feedback and status updates
        - Comprehensive error handling and recovery
        """
        logger.log_info('TelloWaypointNavCoordinator', 'NAVIGATION MODE ACTIVATED')
        
        history = []

        if self.is_connected or self.is_flying:
            logger.log_warning('TelloWaypointNavCoordinator', 'Drone is already connected or flying. Please land it first.')
            return []
        
         # Connect and takeoff
        if not self.connect_drone():
            logger.log_error('TelloWaypointNavCoordinator', 'Failed to connect to drone. Exiting...')
            return []
        
        if not self.takeoff():
            logger.log_error('TelloWaypointNavCoordinator', 'Failed to take off. Exiting...')
            return []
        
        # Initialize navigation interface based on OS
        current_os = platform.system()
        if current_os == 'Windows':
            logger.log_info('TelloWaypointNavCoordinator', 'Detected Windows OS - using Windows navigation interface')
            self.nav_interface = NavigationInterfaceWindows(self.waypoint_dir, self.vertical_factor, self.navigation_speed, self.waypoint_file)
        else:
            logger.log_info('TelloWaypointNavCoordinator', f'Detected {current_os} OS - using Linux navigation interface')
            self.nav_interface = NavigationInterface(self.waypoint_dir, self.vertical_factor, self.navigation_speed, self.waypoint_file)
        
        self.is_navigation_mode = True
        self.is_running = True
        
        try: 
            history = self.nav_interface.run(drone_instance=self.tello)
        except Exception as e:
            logger.log_error('TelloWaypointNavCoordinator', f'Error during navigation: {e}')
        finally: 
            self.is_navigation_mode = False
            self.is_running = False
            return history  # Return navigation history
    
    def run_goto_mode(self): 
        """
        Execute goto mode for direct navigation to specific waypoint destinations.
        
        Performs autonomous navigation to a specified waypoint with configurable
        post-arrival behavior. Includes comprehensive safety monitoring with
        emergency shutdown capabilities and battery management integration.
        
        Workflow:
        1. Emergency shutdown status validation
        2. Drone connection and takeoff sequence
        3. Battery monitoring initialization
        4. Navigation manager setup and waypoint file loading
        5. Target waypoint validation and navigation execution
        6. Post-arrival instruction processing (continue/halt)
        
        Safety Features:
        - Emergency shutdown detection at multiple checkpoints
        - Continuous battery monitoring during navigation
        - Automatic emergency procedures for critical situations
        - Graceful fallback for navigation failures
        
        Returns:
            list: [land_flag, current_waypoint]
                   - land_flag (bool): Whether drone should land after operation
                   - current_waypoint (str): Current or target waypoint ID
                   
        Emergency Handling:
        - Pre-flight emergency status checks
        - Mid-flight emergency response coordination
        - Automatic cleanup and resource deallocation
        """
        logger.log_info('TelloWaypointNavCoordinator', 'GOTO MODE ACTIVATED')
        
        try:
            # Check for emergency shutdown at the start
            if TelloWaypointNavCoordinator._emergency_shutdown:
                return self._goto_mode_emergency_shutdown()

            if not self.is_connected: 
                if not self.connect_drone():
                    logger.log_error('TelloWaypointNavCoordinator', 'Failed to connect to drone. Exiting...')
                    return self._stop_goto_mode()
            
            if not self.is_flying:
                if not self.takeoff():
                    logger.log_error('TelloWaypointNavCoordinator', 'Failed to take off. Exiting...')
                    return self._stop_goto_mode()
            
            # Start battery monitoring if not already running
            if not TelloWaypointNavCoordinator._battery_thread_running:
                self._start_battery_monitoring()

            self.is_goto_mode = True
            self.is_running = True
            
            # Check for emergency shutdown after battery monitoring start
            if TelloWaypointNavCoordinator._emergency_shutdown:
                return self._goto_mode_emergency_shutdown()
            
            if not hasattr(self, 'nav_manager'):
                from .waypoint_navigation import WaypointNavigationManager
                self.nav_manager = WaypointNavigationManager(nav_speed=self.navigation_speed, vertical_factor=self.vertical_factor)
                self.nav_manager.coordinator = self  # Set coordinator reference for emergency shutdown

                # Check if specific waypoint_file is specified
                selected_file = None
                if self.waypoint_file is not None:
                    # Construct the full path to the specified waypoint file
                    specified_file_path = os.path.join(self.waypoint_dir, self.waypoint_file)
                    
                    # Check if the specified file exists
                    if os.path.exists(specified_file_path):
                        logger.log_info('TelloWaypointNavCoordinator', f'Found specified waypoint file: {specified_file_path}')
                        selected_file = specified_file_path
                    else:
                        logger.log_warning('TelloWaypointNavCoordinator', f'Specified waypoint file not found: {specified_file_path}, using latest file.')
                        self.waypoint_file = None  # Reset

                # If no specific file or file not found, use latest file
                if selected_file is None:
                    waypoint_files = self._find_waypoint_files()
                    if not waypoint_files:
                        logger.log_error('TelloWaypointNavCoordinator', 'No waypoint files found. Please run mapping mode first.')
                        return self._stop_goto_mode()
                    
                    selected_file = waypoint_files[0]  # Latest file
                    logger.log_info('TelloWaypointNavCoordinator', f'Using latest waypoint file: {selected_file}')

                if not self.nav_manager.load_waypoint_file(selected_file):
                    logger.log_error('TelloWaypointNavCoordinator', f'Failed to load waypoint file: {selected_file}')
                    return self._stop_goto_mode()
                
                self.current_waypoint = "WP_001"  
                self.nav_manager.current_waypoint_id = self.current_waypoint
            
            if not self.waypoint_dest.startswith("WP_"):
                # Convert to waypoint ID if necessary
                for wp_id, waypoint in self.nav_manager.waypoints.items():
                    if waypoint.name.lower() == self.waypoint_dest.lower():
                        self.waypoint_dest = wp_id
                        break
            
            if self.waypoint_dest not in self.nav_manager.waypoints:
                logger.log_error('TelloWaypointNavCoordinator', f'Waypoint "{self.waypoint_dest}" not found')
                return self._execute_instruction()

            logger.log_info('TelloWaypointNavCoordinator', f'Navigating to waypoint: {self.waypoint_dest}')
            # Check for emergency shutdown before navigation
            if TelloWaypointNavCoordinator._emergency_shutdown:
                return self._goto_mode_emergency_shutdown()

            success = self.nav_manager.navigate_to_waypoint(self.waypoint_dest, self.tello)
        
            if success:
                # Update current waypoint after navigation
                self.current_waypoint = self.waypoint_dest
                logger.log_success('TelloWaypointNavCoordinator', f'Reached waypoint "{self.current_waypoint}"')
                return self._execute_instruction()
            else:
                logger.log_error('TelloWaypointNavCoordinator', f'Failed to reach waypoint "{self.waypoint_dest}"')
                return self._stop_goto_mode()
                
        except Exception as e:
            logger.log_error('TelloWaypointNavCoordinator', f'Error in goto mode: {e}')
            return self._stop_goto_mode()
        finally: 
            self.is_goto_mode = False
            self.is_running = False
    
    def _execute_instruction(self): 
        """
        Process post-navigation instruction to determine operation continuation.
        
        Evaluates the configured NavigationInstruction to determine whether the
        drone should land and terminate the session or continue flying for
        additional operations after reaching the target waypoint.
        
        Instruction Processing:
        - HALT: Initiate landing sequence and session termination
        - CONTINUE: Maintain flight status and preserve active session
        
        Returns:
            tuple: Instruction-specific result from helper methods
                   - HALT: Result from _stop_goto_mode() with cleanup
                   - CONTINUE: Result from _continue_goto_mode() with session preservation
                   
        Emergency Integration:
        - Emergency shutdown status validation before continuation
        - Automatic emergency response overriding normal instruction flow
        """
        if self.instruction == NavigationInstruction.HALT:
            logger.log_info('TelloWaypointNavCoordinator', f'Stopping at waypoint "{self.current_waypoint}"')
            return self._stop_goto_mode()
        else:
            if TelloWaypointNavCoordinator._emergency_shutdown:
                return self._goto_mode_emergency_shutdown()

            logger.log_info('TelloWaypointNavCoordinator', f'Continuing at waypoint "{self.current_waypoint}"')
            # Keep battery monitoring running
            return self._continue_goto_mode()

    def _stop_goto_mode(self): 
        """
        Terminate goto mode operation with complete cleanup and resource deallocation.
        
        Performs comprehensive shutdown sequence including battery monitoring
        termination, singleton instance cleanup, and session state reset.
        Designed for safe operation termination after waypoint arrival or error conditions.
        
        Cleanup Sequence:
        1. Stop background battery monitoring thread
        2. Clear active singleton instance reference
        3. Return termination status with current waypoint information
        
        Returns:
            list: [True, current_waypoint] indicating landing required and final position
        """
        self._stop_battery_monitoring()
        TelloWaypointNavCoordinator._active_instance = None
        return [True, self.current_waypoint]
    
    def _continue_goto_mode(self):
        """
        Continue goto mode operation while maintaining session and monitoring systems.
        
        Preserves active session state and maintains singleton instance reference
        for continued operations. Battery monitoring remains active for ongoing
        safety oversight during extended flight sessions.
        
        Session Preservation:
        1. Maintain active singleton instance reference
        2. Keep battery monitoring thread active
        3. Return continuation status with current waypoint information
        
        Returns:
            list: [False, current_waypoint] indicating no landing required and current position
        """
        TelloWaypointNavCoordinator._active_instance = self
        return [False, self.current_waypoint]
    
    def _goto_mode_emergency_shutdown(self):
        """
        Handle emergency shutdown condition during goto mode operations.
        
        Provides immediate response to emergency shutdown signals with minimal
        processing overhead. Returns landing instruction to ensure safe drone
        recovery during critical situations.
        
        Returns:
            list: [True, current_waypoint] forcing immediate landing and cleanup
        """
        logger.log_warning('TelloWaypointNavCoordinator', 'Emergency shutdown detected - aborting goto mode')
        return [True, self.current_waypoint]
            
    def _find_waypoint_files(self) -> list:
        """
        Discover and return available waypoint files in the configured directory.
        
        Searches the waypoint directory for JSON files matching the standard
        naming convention and returns them sorted by creation time (newest first).
        Used for automatic waypoint file selection in navigation and goto modes.
        
        File Pattern:
        - Searches for "drone_movements_*.json" pattern
        - Sorts results in reverse chronological order
        - Returns empty list if no matching files found
        
        Returns:
            list: Sorted list of waypoint file paths, newest first
            
        Usage:
        - Navigation mode: File selection menu population
        - Goto mode: Automatic latest file selection when no specific file specified
        """
        pattern = os.path.join(self.waypoint_dir, "drone_movements_*.json")
        files = glob.glob(pattern)
        return sorted(files, reverse=True)  # Newest first
        
    def display_controls(self):
        """
        Display comprehensive control instructions and operational guidance for mapping mode.
        """
        logger.log_info('TelloWaypointNavCoordinator', 'Displaying control instructions to user.')
        print("\n" + "="*50)
        print("REAL-TIME DRONE CONTROL")
        print("="*50)
        print("MOVEMENT CONTROLS:")
        print("  W Key          - Move Forward")
        print("  S Key          - Move Backward")
        print("  A Key          - Move Left")
        print("  D Key          - Move Right")
        print("  ↑ Arrow Key    - Move Up")
        print("  ↓ Arrow Key    - Move Down") 
        print("  ← Arrow Key    - Rotate Left (Anticlockwise)")
        print("  → Arrow Key    - Rotate Right (Clockwise)")
        print("\nWAYPOINT CONTROLS:")
        print("  X Key          - Mark Waypoint")
        print("  q Key        - Finish & Land")
        print("\nVIDEO STREAM:")
        print("  📹 Camera view will open in separate window")
        print("  - Live video feed from drone camera")
        print("  - Window will close automatically when mapping ends")
        print("\nNOTES:")
        print("- Hold key to move, release to stop")
        print("- Only one movement/action at a time")
        print("- All movements are recorded automatically")
        print("- Keep video window visible to see drone's perspective")
        print("="*50)
        print()

    def connect_drone(self):
        """
        Establish connection to DJI Tello drone with configuration and status validation.
        """
        try:
            logger.log_info('TelloWaypointNavCoordinator', 'Connecting to Tello drone...')
            self.tello.RESPONSE_TIMEOUT = 7
            self.tello.connect(wait_for_state=False)
            logger.log_success('TelloWaypointNavCoordinator', 'Drone connected successfully!')

            self.is_connected = True

            try:
                battery_response = self.tello.send_command_with_return("battery?", timeout=5)
                logger.log_info('TelloWaypointNavCoordinator', f'Battery: {battery_response}%')
            except Exception as e:
                logger.log_error('TelloWaypointNavCoordinator', f'Battery command failed: {e}')
            
            return True
        except Exception as e:
            logger.log_error('TelloWaypointNavCoordinator', f'Failed to connect to drone: {e}')
            return False
    
    def takeoff(self):
        """
        Execute drone takeoff sequence with safety validation and stabilization.
       """
        if not self.is_connected:
            logger.log_error('TelloWaypointNavCoordinator', 'Drone not connected!')
            return False
            
        try:
            logger.log_info('TelloWaypointNavCoordinator', 'Taking off...')
            self.tello.takeoff()
            self.is_flying = True
            time.sleep(1)  # Wait for stabilization
            logger.log_success('TelloWaypointNavCoordinator', 'Drone is airborne!')
            return True
        
        except Exception as e:
            logger.log_error('TelloWaypointNavCoordinator', f'Takeoff failed: {e}')
            return False
    
    def land(self):
        """
        Execute safe drone landing sequence with status management.
        """
        if self.is_flying:
            try:
                logger.log_info('TelloWaypointNavCoordinator', 'Landing drone...')
                self.tello.land()
                self.is_flying = False
                logger.log_success('TelloWaypointNavCoordinator', 'Drone landed successfully!')
            except Exception as e:
                logger.log_error('TelloWaypointNavCoordinator', f'Landing failed: {e}')
    
    def cleanup(self):
        """
        Perform comprehensive resource cleanup and safe application shutdown.
        
        Orchestrates complete system shutdown with proper resource deallocation,
        including battery monitoring termination, drone landing, connection cleanup,
        and singleton instance management. Ensures safe application exit regardless
        of current operational state.
        
        Cleanup Sequence:
        1. Stop background battery monitoring thread
        2. Land drone if currently flying
        3. Disconnect from drone if connected
        4. Clear singleton instance reference
        5. Log successful cleanup completion
        
        Error Handling:
        - Individual error handling for each cleanup step
        - Non-blocking error recovery to ensure complete cleanup
        - Comprehensive error logging for troubleshooting
        
        Safety Features:
        - Guaranteed drone landing attempt during cleanup
        - Resource deallocation even with individual step failures
        - Singleton instance cleanup for proper lifecycle management
        """
        logger.log_info('TelloWaypointNavCoordinator', 'Cleaning up resources...')

        # Stop battery monitoring if running
        self._stop_battery_monitoring()

        if self.is_flying:
            try: 
                logger.log_info('TelloWaypointNavCoordinator', 'Landing drone during cleanup...')
                self.tello.land()
                self.is_flying = False
            except Exception as e:
                logger.log_error('TelloWaypointNavCoordinator', f'Error during landing: {e}')
        
        if self.is_connected:
            try:
                logger.log_info('TelloWaypointNavCoordinator', 'Disconnecting from drone...')
                self.tello.end()
                self.is_connected = False
            except Exception as e:
                logger.log_error('TelloWaypointNavCoordinator', f'Error during disconnection: {e}')
        
        if TelloWaypointNavCoordinator._active_instance is not None:
            TelloWaypointNavCoordinator._active_instance = None

        logger.log_success('TelloWaypointNavCoordinator', 'Application closed successfully')

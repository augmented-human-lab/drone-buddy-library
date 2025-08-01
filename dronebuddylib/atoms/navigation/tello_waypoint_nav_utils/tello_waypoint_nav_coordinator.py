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
        Starts background battery monitoring thread for continuous safety oversight.
        
        Creates a daemon thread that monitors battery level and triggers emergency 
        landing if battery drops below 10% to prevent drone loss.
        """
        if not TelloWaypointNavCoordinator._battery_thread_running:
            TelloWaypointNavCoordinator._battery_thread_running = True
            TelloWaypointNavCoordinator._battery_thread = threading.Thread(target=self._battery_monitor_loop, daemon=True)
            TelloWaypointNavCoordinator._battery_thread.start()
            logger.log_info('TelloWaypointNavCoordinator', 'Battery monitoring thread started.')
    
    def _stop_battery_monitoring(self):
        """
        Stops background battery monitoring thread with graceful cleanup.
        
        Safely terminates the battery monitoring thread with 2-second timeout.
        """
        if TelloWaypointNavCoordinator._battery_thread_running:
            TelloWaypointNavCoordinator._battery_thread_running = False
            if TelloWaypointNavCoordinator._battery_thread and TelloWaypointNavCoordinator._battery_thread.is_alive():
                TelloWaypointNavCoordinator._battery_thread.join(timeout=2)
            logger.log_info('TelloWaypointNavCoordinator', 'Battery monitoring thread stopped.')
    
    @classmethod
    def _pause_battery_monitoring(cls):
        """
        Temporarily pauses battery monitoring during navigation operations.
        
        Prevents battery monitoring interference during critical flight maneuvers.
        """
        cls._battery_monitoring_paused = True
        logger.log_debug('TelloWaypointNavCoordinator', 'Battery monitoring paused.')
    
    @classmethod
    def _resume_battery_monitoring(cls):
        """
        Resumes battery monitoring after navigation operations complete.
        
        Re-enables continuous battery monitoring for ongoing safety oversight.
        """
        cls._battery_monitoring_paused = False
        logger.log_debug('TelloWaypointNavCoordinator', 'Battery monitoring resumed.')
    
    def _battery_monitor_loop(self):
        """
        Background battery monitoring loop with emergency safety management.
        
        Runs continuously monitoring battery level every 5 seconds. Automatically 
        triggers emergency landing and program termination if battery drops below 10%.
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
        Executes the main application workflow based on configured operational mode.
        
        Orchestrates mode-specific execution (mapping/navigation/goto) with comprehensive 
        error handling and guaranteed cleanup regardless of operation outcome.
        
        Returns:
            list: Mode-specific execution summary
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
        Executes mapping mode for waypoint creation through manual drone control.
        
        Provides real-time manual control interface allowing users to fly the drone 
        and create waypoint maps. Uses platform-specific controllers for Windows/Linux.
        
        Returns:
            list: Summary of waypoints created during mapping session
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
        Executes interactive navigation mode for waypoint-based autonomous flight.
        
        Provides user interface for selecting waypoints from existing maps and executing 
        autonomous navigation between selected points.
        
        Returns:
            list: Navigation history including visited waypoints
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
        Executes goto mode for direct navigation to specific waypoint destinations.
        
        Performs autonomous navigation to a specified waypoint with configurable 
        post-arrival behavior. Includes emergency shutdown detection and battery monitoring.
        
        Returns:
            list: [land_flag, current_waypoint] indicating landing status and position
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
                    self.waypoint_file = os.path.basename(selected_file)  # Store file name for reference
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
        Processes post-navigation instruction to determine operation continuation.
        
        Evaluates NavigationInstruction to determine if drone should land (HALT) 
        or continue flying (CONTINUE) after reaching target waypoint.
        
        Returns:
            tuple: Instruction-specific result with landing status
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
        Terminates goto mode operation with complete cleanup.
        
        Stops battery monitoring, clears singleton instance, and returns 
        landing instruction with current waypoint information.
        
        Returns:
            list: [True, current_waypoint] indicating landing required
        """
        self._stop_battery_monitoring()
        TelloWaypointNavCoordinator._active_instance = None
        return [True, self.current_waypoint]
    
    def _continue_goto_mode(self):
        """
        Continues goto mode operation while maintaining session.
        
        Preserves active session state and maintains singleton instance 
        for continued operations with ongoing battery monitoring.
        
        Returns:
            list: [False, current_waypoint] indicating no landing required
        """
        TelloWaypointNavCoordinator._active_instance = self
        return [False, self.current_waypoint]
    
    def _goto_mode_emergency_shutdown(self):
        """
        Handles emergency shutdown condition during goto mode operations.
        
        Provides immediate response to emergency shutdown signals with 
        minimal processing overhead for safe drone recovery.
        
        Returns:
            list: [True, current_waypoint] forcing immediate landing
        """
        logger.log_warning('TelloWaypointNavCoordinator', 'Emergency shutdown detected - aborting goto mode')
        return [True, self.current_waypoint]
            
    def _find_waypoint_files(self) -> list:
        """
        Discovers and returns available waypoint files in the configured directory.
        
        Searches for JSON files matching "drone_movements_*.json" pattern 
        and returns them sorted newest first.
        
        Returns:
            list: Sorted list of waypoint file paths, newest first
        """
        pattern = os.path.join(self.waypoint_dir, "drone_movements_*.json")
        files = glob.glob(pattern)
        return sorted(files, reverse=True)  # Newest first
        
    def display_controls(self):
        """
        Displays comprehensive control instructions for mapping mode.
        
        Shows formatted console output with movement controls, waypoint controls, 
        video streaming info, and operational guidelines.
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
        Establishes connection to DJI Tello drone with status validation.
        
        Configures response timeout, connects to drone, and retrieves battery status.
        
        Returns:
            bool: True if connection successful, False otherwise
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
        Executes drone takeoff sequence with safety validation.
        
        Validates connection status, performs takeoff, and waits for stabilization.
        
        Returns:
            bool: True if takeoff successful, False otherwise
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
        Executes safe drone landing sequence with status management.
        
        Performs controlled landing operation and updates flight status flag.
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
        Performs comprehensive resource cleanup and safe application shutdown.
        
        Orchestrates complete system shutdown including battery monitoring termination, 
        drone landing, connection cleanup, and singleton instance management.
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

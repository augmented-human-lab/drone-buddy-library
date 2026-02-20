"""
Main coordinator for DJI Tello drone 2D hierarchical waypoint navigation system.

This module serves as the central orchestrator for all drone navigation operations, managing
Super Waypoints and Inner Waypoints with smart routing capabilities.

Operational Modes:
- MAPPING: Manual flight control to create 2D hierarchical waypoint maps
- NAVIGATION: Interactive waypoint selection with smart routing through Super Waypoint hubs
- GOTO: Direct navigation to specific waypoints with automatic hub traversal

Key Features:
- 2D hierarchical waypoint structure (Super Waypoints + Inner Waypoints)
- Smart routing through Super Waypoint hubs
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
    from .waypoint_navigation import WaypointNavigationManager
    from .tello_nav_extra import MiDaSObstacleDetector
    
from dronebuddylib.models.enums import ObstacleDetectionMode
    
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
    Central coordinator for DJI Tello drone 2D hierarchical waypoint navigation system.
    
    This class orchestrates all aspects of 2D drone navigation operations with
    Super Waypoint hubs and Inner Waypoint branches. It implements smart routing
    that automatically navigates through hub points.
    
    The coordinator operates in three distinct modes:
    - MAPPING: Real-time manual control for 2D waypoint creation with Super/Inner waypoints
    - NAVIGATION: Interactive waypoint navigation with user selection interface and smart hub routing
    - GOTO: Direct autonomous navigation to specific waypoints with automatic hub traversal
    """

    # Class-level variables for singleton pattern and safety monitoring
    _active_instance = None               # Singleton instance tracker
    _battery_thread = None               # Background battery monitoring thread
    _battery_thread_running = False      # Control flag for battery monitoring loop
    _battery_monitoring_paused = False   # Pause flag for navigation operations
    _emergency_shutdown = False          # Emergency shutdown trigger for critical situations
    _disable_cv2_video_window = False    # Set True to disable OpenCV window (for GUI integration)
    _external_nav_manager_callback = None # Callback to receive nav_manager for GUI video integration
    _session_terminated = False          # Set True when session ends (obstacle timeout, keyboard interrupt) - prevents further navigation
    _obstacle_timeout_occurred = False   # Set True specifically when obstacle timeout (30s) caused termination

    @classmethod
    def get_instance(cls, waypoint_dir: str, vertical_factor: float, movement_speed: int, 
                     rotation_speed: int, navigation_speed: int, mode: str, 
                     waypoint_dest: str = None, instruction: NavigationInstruction = None, 
                     waypoint_file: str = None, create_new: bool = False,
                     obstacle_detection_mode: 'ObstacleDetectionMode' = None,
                     midas_model_path: str = None):
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
            obstacle_detection_mode (ObstacleDetectionMode, optional): Depth-based obstacle detection threshold
            midas_model_path (str, optional): Path to MiDaS ONNX model file
        
        Returns:
            TelloWaypointNavCoordinator: Configured singleton coordinator instance
            
        Raises:
            ValueError: For invalid mode specifications or parameter combinations
        """
        if create_new: 
            instance = cls(waypoint_dir, vertical_factor, movement_speed, rotation_speed, 
                          navigation_speed, mode, waypoint_dest, instruction, waypoint_file,
                          obstacle_detection_mode, midas_model_path)
            cls._active_instance = instance
            return instance
        else: 
            # Update parameters of existing instance
            instance = cls._active_instance
            instance.waypoint_dest = waypoint_dest
            instance.instruction = instruction
            return instance 

    def __init__(self, waypoint_dir: str, vertical_factor: float, movement_speed: int, 
                 rotation_speed: int, navigation_speed: int, mode: str, 
                 waypoint_dest: str = None, instruction: NavigationInstruction = None, 
                 waypoint_file: str = None, obstacle_detection_mode: 'ObstacleDetectionMode' = None,
                 midas_model_path: str = None):
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
            obstacle_detection_mode (ObstacleDetectionMode, optional): Depth-based obstacle detection threshold
            midas_model_path (str, optional): Path to MiDaS ONNX model file for depth estimation
            
        Raises:
            ValueError: For invalid operational mode or incompatible parameter combinations
            OSError: For waypoint directory access issues or platform compatibility problems
            
        Note:
            Drone connection is deferred until explicit connect_drone() call for resource management
        """
        logger.log_info('TelloWaypointNavCoordinator', f'Initializing coordinator in {mode} mode.')
        
        # Store configuration parameters
        self.waypoint_dir = waypoint_dir
        self.vertical_factor = vertical_factor
        self.movement_speed = movement_speed
        self.rotation_speed = rotation_speed
        self.navigation_speed = navigation_speed
        self.mode = mode
        self.waypoint_dest = waypoint_dest
        self.instruction = instruction
        self.waypoint_file = waypoint_file
        
        # Obstacle detection configuration
        self.obstacle_detection_mode = obstacle_detection_mode
        self.midas_model_path = midas_model_path
        self.obstacle_detector = None  # Created when video stream is started
        self.frame_read = None         # Drone camera frame reader

        # Initialize Tello drone
        logger.log_debug('TelloWaypointNavCoordinator', 'Initializing Tello drone.')
        self.tello = Tello()

        # Initialize operation state flags
        self.is_connected = False
        self.is_flying = False
        self.is_mapping_mode = False
        self.is_navigation_mode = False
        self.is_goto_mode = False
        self.is_running = False
        self.current_waypoint = "SWP_001"  # Start at first Super Waypoint
        self._video_stream_active = False  # Track video stream state

        # Always reset the class-level variables when new instance is created
        TelloWaypointNavCoordinator._emergency_shutdown = False
        TelloWaypointNavCoordinator._battery_thread_running = False
        TelloWaypointNavCoordinator._battery_monitoring_paused = False
        TelloWaypointNavCoordinator._active_instance = None
        TelloWaypointNavCoordinator._battery_thread = None
        # NOTE: Do NOT reset _disable_cv2_video_window here - it's set by GUI before navigation starts
        
        logger.log_debug('TelloWaypointNavCoordinator', 
            f'Coordinator initialized: mode={mode}, waypoint_dir={waypoint_dir}, '
            f'obstacle_detection={obstacle_detection_mode}')
    
    def _start_battery_monitoring(self):
        """
        Starts background battery monitoring thread for continuous safety oversight.
        
        Creates a daemon thread that monitors battery level and triggers emergency 
        landing if battery drops below 10% to prevent drone loss.
        """
        if not TelloWaypointNavCoordinator._battery_thread_running:
            TelloWaypointNavCoordinator._battery_thread_running = True
            TelloWaypointNavCoordinator._battery_thread = threading.Thread(
                target=self._battery_monitor_loop, daemon=True)
            TelloWaypointNavCoordinator._battery_thread.start()
            logger.log_info('TelloWaypointNavCoordinator', 'Battery monitoring thread started.')
    
    def _stop_battery_monitoring(self):
        """
        Stops background battery monitoring thread with graceful cleanup.
        
        Safely terminates the battery monitoring thread with 2-second timeout.
        """
        if TelloWaypointNavCoordinator._battery_thread_running:
            TelloWaypointNavCoordinator._battery_thread_running = False
            if (TelloWaypointNavCoordinator._battery_thread and 
                TelloWaypointNavCoordinator._battery_thread.is_alive()):
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
                        logger.log_error('TelloWaypointNavCoordinator', 
                            f'CRITICAL: Battery too low ({battery}%), initiating emergency landing.')
                        # Trigger emergency shutdown sequence
                        TelloWaypointNavCoordinator._emergency_shutdown = True
                        # Stop all ongoing operations
                        instance.is_running = False
                        instance.is_goto_mode = False
                        
                        # Emergency stop and landing sequence
                        try:
                            instance.tello.send_rc_control(0, 0, 0, 0)  # Stop immediately
                            time.sleep(0.5)  # Brief pause to ensure stop command is processed

                            if instance.is_flying:
                                logger.log_info('TelloWaypointNavCoordinator', 
                                    'Landing drone due to critical battery level.')
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
                        
                        # Complete cleanup and force exit
                        if TelloWaypointNavCoordinator._active_instance is not None:
                            TelloWaypointNavCoordinator._active_instance = None
                        # Stop battery monitoring
                        TelloWaypointNavCoordinator._battery_thread_running = False
                        logger.log_error('TelloWaypointNavCoordinator', 
                            'EMERGENCY SHUTDOWN: Program terminating due to critical battery level')
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
            
        Raises:
            KeyboardInterrupt: Re-raised after cleanup to allow calling code to handle it
        """
        logger.log_info('TelloWaypointNavCoordinator', f'Starting navigation run in {self.mode} mode.')
        
        summary = []
        land = True  # Default to landing on completion
        interrupted = False  # Track if user interrupted
        try:
            if self.mode == "mapping":
                summary = self._run_mapping_mode()
            elif self.mode == "navigation":
                summary = self._run_navigation_mode()
            elif self.mode == "goto":
                summary = self.run_goto_mode()
                land = summary[0]  # Extract land flag from goto mode result
            
        except KeyboardInterrupt:
            logger.log_warning('TelloWaypointNavCoordinator', 'Application interrupted by user.')
            land = True  # Force landing on user interrupt
            interrupted = True  # Mark as interrupted
            TelloWaypointNavCoordinator._session_terminated = True  # Prevent further navigation
            summary = [True, self.current_waypoint]  # Return proper result format
        except Exception as e:
            logger.log_error('TelloWaypointNavCoordinator', f'Application error: {e}')
            import traceback
            traceback.print_exc()
            land = True  # Force landing on error
            TelloWaypointNavCoordinator._session_terminated = True  # Prevent further navigation
            summary = [True, self.current_waypoint]  # Return proper result format
        finally:
            if land: 
                self.is_running = False
                self.is_mapping_mode = False
                self.is_navigation_mode = False
                self._stop_battery_monitoring()  # Safe cleanup - only stops if running
                TelloWaypointNavCoordinator._active_instance = None
                self.cleanup()
            
            # Re-raise KeyboardInterrupt AFTER cleanup so calling code stops
            if interrupted:
                raise KeyboardInterrupt("User interrupted navigation - drone landed safely")

            return summary

    def _run_mapping_mode(self) -> list:
        """
        Executes 2D mapping mode for hierarchical waypoint creation.
        
        Provides real-time manual control interface allowing users to fly the drone 
        and create 2D hierarchical waypoint maps with Super Waypoints and Inner Waypoints.
        Uses platform-specific controllers for Windows/Linux.
        
        Returns:
            list: Summary of waypoints created during mapping session
        """
        logger.log_info('TelloWaypointNavCoordinator', '2D MAPPING MODE ACTIVATED')
        
        print("\n" + "=" * 60)
        print("🗺️  2D HIERARCHICAL WAYPOINT MAPPING MODE")
        print("=" * 60)
        print("\nYou will create a web of interconnected waypoints:")
        print("- ⭐ Super Waypoints: Hub points connecting different areas")
        print("- 📍 Inner Waypoints: Local points branching off hubs")
        print("\nThe drone automatically returns to Super Waypoints after")
        print("marking inner waypoints, creating a navigable network.")
        print("=" * 60)

        if self.is_connected or self.is_flying:
            logger.log_warning('TelloWaypointNavCoordinator', 
                'Drone is already connected or flying. Please land it first.')
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
            logger.log_info('TelloWaypointNavCoordinator', 
                'Detected Windows OS - using Windows controller with video streaming')
            self.drone_controller = RealTimeDroneControllerWindows(
                self.waypoint_dir, 
                self.movement_speed, 
                self.rotation_speed,
                self.navigation_speed,
                self.vertical_factor
            )
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
            import traceback
            traceback.print_exc()
            summary = []
        finally:
            self.is_mapping_mode = False
            self.is_running = False

            return summary  # Return summary of waypoints created
    
    def _run_navigation_mode(self) -> list:
        """
        Executes 2D navigation mode with smart hub routing.
        
        Provides user interface for selecting waypoints from existing 2D maps and executing 
        autonomous navigation between selected points with smart routing through Super Waypoint hubs.
        
        Returns:
            list: Navigation history including visited waypoints
        """
        logger.log_info('TelloWaypointNavCoordinator', '2D NAVIGATION MODE ACTIVATED')
        
        history = []

        if self.is_connected or self.is_flying:
            logger.log_warning('TelloWaypointNavCoordinator', 
                'Drone is already connected or flying. Please land it first.')
            return []
        
        # Connect and takeoff
        if not self.connect_drone():
            logger.log_error('TelloWaypointNavCoordinator', 'Failed to connect to drone. Exiting...')
            return []
        
        if not self.takeoff():
            logger.log_error('TelloWaypointNavCoordinator', 'Failed to take off. Exiting...')
            return []
        
        # Start video stream for obstacle detection and visual feedback
        if not self.start_video_stream():
            logger.log_warning('TelloWaypointNavCoordinator', 
                'Video stream failed to start - continuing without obstacle detection')
        
        # Select platform-specific navigation interface
        current_os = platform.system()
        if current_os == 'Windows':
            logger.log_info('TelloWaypointNavCoordinator', 
                'Detected Windows OS - using Windows navigation interface')
            self.nav_interface = NavigationInterfaceWindows(
                self.waypoint_dir, 
                self.vertical_factor, 
                self.navigation_speed, 
                self.waypoint_file,
                obstacle_detector=self.obstacle_detector,
                frame_read=self.frame_read
            )
        else:
            logger.log_info('TelloWaypointNavCoordinator', f'Detected {current_os} OS - using Linux navigation interface')
            self.nav_interface = NavigationInterface(self.waypoint_dir, self.vertical_factor, self.navigation_speed, self.waypoint_file)
        
        self.is_navigation_mode = True
        self.is_running = True
        
        # Start video display if frame_read is available
        # The display loop handles GUI vs cv2 mode automatically via callback
        if self.frame_read and hasattr(self.nav_interface, 'nav_manager'):
            # Notify external callback if GUI integration is set up
            if TelloWaypointNavCoordinator._external_nav_manager_callback:
                try:
                    TelloWaypointNavCoordinator._external_nav_manager_callback(self.nav_interface.nav_manager)
                except Exception as e:
                    logger.log_warning('TelloWaypointNavCoordinator', 
                        f'External nav_manager callback failed: {e}')
            self.nav_interface.nav_manager.start_video_display(window_name="Tello Navigation View")
        
        try: 
            history = self.nav_interface.run(drone_instance=self.tello)
        except Exception as e:
            logger.log_error('TelloWaypointNavCoordinator', f'Error during navigation: {e}')
        finally:
            # Stop video display if running
            if hasattr(self, 'nav_interface') and hasattr(self.nav_interface, 'nav_manager'):
                self.nav_interface.nav_manager.stop_video_display()
            self.is_navigation_mode = False
            self.is_running = False
            return history  # Return navigation history
    
    def run_goto_mode(self): 
        """
        Executes 2D goto mode with smart hub traversal.
        
        Performs autonomous navigation to a specified waypoint with configurable 
        post-arrival behavior. Includes emergency shutdown detection and battery monitoring.
        Smart routing automatically navigates through Super Waypoint hubs.
        
        Returns:
            list: [land_flag, current_waypoint] indicating landing status and position
        """
        logger.log_info('TelloWaypointNavCoordinator', '2D GOTO MODE ACTIVATED')
        
        try:
            # Check if session was terminated (obstacle timeout, keyboard interrupt, etc.)
            # This prevents the drone from taking off again after a safety landing
            logger.log_debug('TelloWaypointNavCoordinator', 
                f'Session terminated flag check: _session_terminated={TelloWaypointNavCoordinator._session_terminated}')
            if TelloWaypointNavCoordinator._session_terminated:
                logger.log_warning('TelloWaypointNavCoordinator', 
                    'Session terminated - refusing to start new navigation (drone already landed for safety)')
                return [True, self.current_waypoint]  # Return landed status
            
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
            
            # Start video stream for obstacle detection and visual feedback
            if not self.start_video_stream():
                logger.log_warning('TelloWaypointNavCoordinator', 
                    'Video stream failed to start - continuing without obstacle detection')

            self.is_goto_mode = True
            self.is_running = True
            
            # Check for emergency shutdown after battery monitoring start
            if TelloWaypointNavCoordinator._emergency_shutdown:
                return self._goto_mode_emergency_shutdown()
            
            if not hasattr(self, 'nav_manager'):
                self.nav_manager = WaypointNavigationManager(
                    nav_speed=self.navigation_speed, 
                    vertical_factor=self.vertical_factor,
                    obstacle_detector=self.obstacle_detector,
                    frame_read=self.frame_read
                )
                self.nav_manager.coordinator = self  # Enable emergency shutdown callbacks
                
                # Notify external callback if GUI integration is set up
                if TelloWaypointNavCoordinator._external_nav_manager_callback:
                    try:
                        TelloWaypointNavCoordinator._external_nav_manager_callback(self.nav_manager)
                    except Exception as e:
                        logger.log_warning('TelloWaypointNavCoordinator', 
                            f'External nav_manager callback failed: {e}')
                
                # Start video display for visual feedback during navigation
                # The display loop handles GUI vs cv2 mode automatically via callback
                if self.frame_read:
                    self.nav_manager.start_video_display(window_name="Tello Navigation View")

                # Check if specific waypoint_file is specified
                selected_file = None
                if self.waypoint_file is not None:
                    # Construct the full path to the specified waypoint file
                    specified_file_path = os.path.join(self.waypoint_dir, self.waypoint_file)
                    
                    # Check if the specified file exists
                    if os.path.exists(specified_file_path):
                        logger.log_info('TelloWaypointNavCoordinator', 
                            f'Found specified waypoint file: {specified_file_path}')
                        selected_file = specified_file_path
                    else:
                        logger.log_warning('TelloWaypointNavCoordinator', 
                            f'Specified waypoint file not found: {specified_file_path}, using latest file.')
                        self.waypoint_file = None  # Reset for fallback

                # Fallback to latest available file
                if selected_file is None:
                    waypoint_files = self._find_waypoint_files()
                    if not waypoint_files:
                        logger.log_error('TelloWaypointNavCoordinator', 
                            'No waypoint files found. Please run mapping mode first.')
                        return self._stop_goto_mode()
                    
                    selected_file = waypoint_files[0]  # Latest file
                    self.waypoint_file = os.path.basename(selected_file)  # Store file name for reference
                    logger.log_info('TelloWaypointNavCoordinator', f'Using latest waypoint file: {selected_file}')

                if not self.nav_manager.load_waypoint_file(selected_file):
                    logger.log_error('TelloWaypointNavCoordinator', f'Failed to load waypoint file: {selected_file}')
                    return self._stop_goto_mode()
                
                self.current_waypoint = self.nav_manager.current_waypoint_id
            
            # Resolve waypoint destination (handle names and IDs)
            target_id = self._resolve_waypoint_id(self.waypoint_dest)
            
            if target_id is None:
                logger.log_error('TelloWaypointNavCoordinator', 
                    f'Waypoint "{self.waypoint_dest}" not found')
                return self._execute_instruction()

            logger.log_info('TelloWaypointNavCoordinator', f'Navigating to waypoint: {target_id}')
            
            # Final emergency check before movement
            if TelloWaypointNavCoordinator._emergency_shutdown:
                return self._goto_mode_emergency_shutdown()

            success = self.nav_manager.navigate_to_waypoint(target_id, self.tello)
        
            if success:
                # Update position after successful navigation
                self.current_waypoint = target_id
                logger.log_success('TelloWaypointNavCoordinator', f'Reached waypoint "{self.current_waypoint}"')
                return self._execute_instruction()
            else:
                logger.log_error('TelloWaypointNavCoordinator', f'Failed to reach waypoint "{target_id}"')
                return self._stop_goto_mode()
                
        except Exception as e:
            logger.log_error('TelloWaypointNavCoordinator', f'Error in goto mode: {e}')
            import traceback
            traceback.print_exc()
            return self._stop_goto_mode()
        finally:
            # Only stop video display if we're landing (HALT instruction)
            # Video display should persist across multiple CONTINUE waypoints
            if self.instruction == NavigationInstruction.HALT:
                if hasattr(self, 'nav_manager') and self.nav_manager:
                    self.nav_manager.stop_video_display()
            self.is_goto_mode = False
            self.is_running = False
    
    def _resolve_waypoint_id(self, waypoint_name_or_id: str) -> str:
        """
        Resolve waypoint name or ID to actual waypoint ID.
        
        Handles both Super Waypoints (SWP_XXX) and Inner Waypoints (SWP_XXX_IWP_XXX)
        by searching through all_waypoints dictionary.
        
        Args:
            waypoint_name_or_id (str): Waypoint name or ID to resolve
            
        Returns:
            str: Resolved waypoint ID or None if not found
        """
        # Check if it's already a valid ID
        if waypoint_name_or_id in self.nav_manager.all_waypoints:
            return waypoint_name_or_id
        
        # Search by name (case-insensitive)
        for wp_id, waypoint in self.nav_manager.all_waypoints.items():
            if waypoint.name.lower() == waypoint_name_or_id.lower():
                return wp_id
        
        return None
    
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
            # Maintain battery monitoring and instance state
            return self._continue_goto_mode()

    def _stop_goto_mode(self): 
        """
        Terminates goto mode operation with complete cleanup.
        
        Stops battery monitoring, clears singleton instance, and returns 
        landing instruction with current waypoint information.
        Sets _session_terminated to prevent any further navigation attempts.
        
        Returns:
            list: [True, current_waypoint] indicating landing required
        """
        self._stop_battery_monitoring()
        TelloWaypointNavCoordinator._active_instance = None
        TelloWaypointNavCoordinator._session_terminated = True  # Prevent further navigation
        logger.log_warning('TelloWaypointNavCoordinator', 
            f'Session terminated flag SET TO TRUE - no further navigation allowed (current_waypoint={self.current_waypoint})')
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
        return sorted(files, reverse=True)  # Sort newest first
        
    def display_controls(self):
        """
        Displays comprehensive control instructions for 2D mapping mode.
        
        Shows formatted console output with movement controls, 2D waypoint controls, 
        video streaming info, and operational guidelines.
        """
        logger.log_info('TelloWaypointNavCoordinator', 'Displaying control instructions to user.')
        print("\n" + "=" * 60)
        print("2D HIERARCHICAL WAYPOINT MAPPING CONTROLS")
        print("=" * 60)
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
        print("  X Key          - Mark Waypoint (choose Super or Inner)")
        print("  Q Key          - Finish & Land")
        print("\n2D WAYPOINT SYSTEM:")
        print("  ⭐ Super Waypoints - Hub points that connect areas")
        print("     Drone stays at Super Waypoints after marking")
        print("  📍 Inner Waypoints - Branch points off Super Waypoints")
        print("     Drone returns to Super Waypoint after marking")
        print("\nVIDEO STREAM:")
        print("  📹 Camera view will open in separate window")
        print("  - Live video feed from drone camera")
        print("  - Window will close automatically when mapping ends")
        print("\nNOTES:")
        print("- Hold key to move, release to stop")
        print("- Only one movement/action at a time")
        print("- All movements are recorded automatically")
        print("- Keep video window visible to see drone's perspective")
        print("=" * 60)
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
            time.sleep(1)  # Stabilization delay
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
    
    # ==================== VIDEO STREAMING MANAGEMENT ====================
    
    def start_video_stream(self):
        """
        Start drone video streaming and initialize obstacle detector if configured.
        
        Initiates video stream from drone, gets frame reader, and creates
        MiDaS obstacle detector if obstacle detection mode is enabled.
        
        Returns:
            bool: True if video stream started successfully, False otherwise
        """
        if self._video_stream_active:
            logger.log_debug('TelloWaypointNavCoordinator', 'Video stream already active')
            return True
        
        try:
            logger.log_info('TelloWaypointNavCoordinator', 'Starting video stream...')
            self.tello.streamon()
            time.sleep(1)  # Allow stream to initialize
            
            self.frame_read = self.tello.get_frame_read()
            self._video_stream_active = True
            
            logger.log_success('TelloWaypointNavCoordinator', 'Video stream started')
            
            # Initialize obstacle detector if mode is enabled
            if self.obstacle_detection_mode and self.obstacle_detection_mode != ObstacleDetectionMode.OFF:
                self._init_obstacle_detector()
            
            return True
            
        except Exception as e:
            logger.log_error('TelloWaypointNavCoordinator', f'Failed to start video stream: {e}')
            return False
    
    def stop_video_stream(self):
        """
        Stop drone video streaming and cleanup obstacle detector.
        
        Turns off video stream and releases associated resources.
        Only call this when ending the navigation session.
        """
        if not self._video_stream_active:
            return
        
        try:
            logger.log_info('TelloWaypointNavCoordinator', 'Stopping video stream...')
            self.tello.streamoff()
            self._video_stream_active = False
            self.frame_read = None
            self.obstacle_detector = None
            logger.log_success('TelloWaypointNavCoordinator', 'Video stream stopped')
        except Exception as e:
            logger.log_error('TelloWaypointNavCoordinator', f'Error stopping video stream: {e}')
    
    def _init_obstacle_detector(self):
        """
        Initialize MiDaS obstacle detector with configured parameters.
        
        Creates MiDaSObstacleDetector instance with the configured detection mode
        and model path. Logs warning if model path not provided or loading fails.
        """
        try:
            if not self.midas_model_path:
                logger.log_warning('TelloWaypointNavCoordinator', 
                    'MiDaS model path not provided - obstacle detection disabled')
                return
            
            logger.log_info('TelloWaypointNavCoordinator', 
                f'Initializing MiDaS obstacle detector (mode: {self.obstacle_detection_mode.name})...')
            
            self.obstacle_detector = MiDaSObstacleDetector(
                model_path=self.midas_model_path,
                detection_mode=self.obstacle_detection_mode
            )
            
            # Initialize the model (load ONNX)
            if not self.obstacle_detector.initialize():
                logger.log_error('TelloWaypointNavCoordinator', 'Failed to initialize MiDaS model')
                self.obstacle_detector = None
                return
            
            logger.log_success('TelloWaypointNavCoordinator', 
                f'Obstacle detector initialized with threshold: {self.obstacle_detection_mode.value}')
                
        except Exception as e:
            logger.log_error('TelloWaypointNavCoordinator', f'Failed to initialize obstacle detector: {e}')
            self.obstacle_detector = None
    
    def cleanup(self):
        """
        Performs comprehensive resource cleanup and safe application shutdown.
        
        Orchestrates complete system shutdown including video stream termination,
        battery monitoring termination, drone landing, connection cleanup, 
        and singleton instance management.
        """
        logger.log_info('TelloWaypointNavCoordinator', 'Cleaning up resources...')

        # Stop video stream if running
        self.stop_video_stream()
        
        # Stop video display if nav_manager has it running
        if hasattr(self, 'nav_manager') and self.nav_manager:
            self.nav_manager.stop_video_display()

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

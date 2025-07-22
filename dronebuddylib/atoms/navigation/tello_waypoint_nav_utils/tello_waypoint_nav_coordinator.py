from djitellopy import Tello
import time
import threading
import os
import glob
import sys
from enum import Enum

from .realtime_drone_control import RealTimeDroneController
from .navigation_interface import NavigationInterface
from dronebuddylib.utils.logger import Logger

logger = Logger()

class NavigationInstruction(Enum):
    """Enumeration for navigation instructions in goto mode."""
    CONTINUE = "continue"
    HALT = "halt"

class TelloWaypointNavCoordinator: 
    """
    Class for handling Tello waypoint mapping on Linux.
    """

    _active_instance = None # Class-level instance tracker
    _battery_thread = None # Background battery monitoring thread
    _battery_thread_running = False # Flag to control battery thread
    _battery_monitoring_paused = False # Flag to pause battery monitoring during navigation
    _emergency_shutdown = False # Flag to trigger emergency program termination

    @classmethod
    def get_instance(cls, waypoint_dir: str, vertical_factor: float, movement_speed: int, rotation_speed: int, navigation_speed: int, mode: str, waypoint_dest: str = None, instruction: NavigationInstruction = None, waypoint_file: str = None, create_new: bool = False):
        """
        Factory method to get the existing or new TelloWaypointNavCoordinator instance.
        
        Args:
            waypoint_dir (str): Directory to save waypoint files.
            vertical_factor (float): Vertical movement factor during navigation.
            movement_speed (int): Speed for mapping movements.
            rotation_speed (int): Speed for rotation during mapping.
            navigation_speed (int): Speed for navigation.
            mode (str): Mode of operation, e.g., 'mapping', 'navigation', 'goto'.
            waypoint_dest (str, optional): Destination waypoint for goto mode.
            instruction (NavigationInstruction, optional): Instruction for goto mode.
            create_new (bool): Whether to create a new instance or return the existing one.
        
        Returns:
            TelloWaypointNavCoordinator instance.
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
        Initializes the TelloWaypointNavCoordinator with the given parameters.

        Args:
            waypoint_dir (str): Directory to save waypoint files.
            vertical_factor (float): Vertical movement factor during navigation.
            movement_speed (int): Speed for mapping movements.
            rotation_speed (int): Speed for rotation during mapping.
            navigation_speed (int): Speed for navigation.
            mode (str): Mode of operation, e.g., 'mapping', 'navigation', 'goto'.
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
        """Start background battery monitoring thread for goto mode."""
        if not TelloWaypointNavCoordinator._battery_thread_running:
            TelloWaypointNavCoordinator._battery_thread_running = True
            TelloWaypointNavCoordinator._battery_thread = threading.Thread(target=self._battery_monitor_loop, daemon=True)
            TelloWaypointNavCoordinator._battery_thread.start()
            logger.log_info('TelloWaypointNavCoordinator', 'Battery monitoring thread started.')
    
    def _stop_battery_monitoring(self):
        """Stop background battery monitoring thread."""
        if TelloWaypointNavCoordinator._battery_thread_running:
            TelloWaypointNavCoordinator._battery_thread_running = False
            if TelloWaypointNavCoordinator._battery_thread and TelloWaypointNavCoordinator._battery_thread.is_alive():
                TelloWaypointNavCoordinator._battery_thread.join(timeout=2)
            logger.log_info('TelloWaypointNavCoordinator', 'Battery monitoring thread stopped.')
    
    @classmethod
    def _pause_battery_monitoring(cls):
        """Pause battery monitoring during navigation operations."""
        cls._battery_monitoring_paused = True
        logger.log_debug('TelloWaypointNavCoordinator', 'Battery monitoring paused.')
    
    @classmethod
    def _resume_battery_monitoring(cls):
        """Resume battery monitoring after navigation operations."""
        cls._battery_monitoring_paused = False
        logger.log_debug('TelloWaypointNavCoordinator', 'Battery monitoring resumed.')
    
    def _battery_monitor_loop(self):
        """Background battery monitoring loop."""
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
        """Run the main application."""
        logger.log_info('TelloWaypointNavCoordinator', f'Starting navigation run in {self.mode} mode.')
        
        summary = []
        land = True  # Default to landing at the end
        try:
            if self.mode == "mapping":
                summary = self._run_mapping_mode()
            elif self.mode == "navigation":
                summary = self._run_navigation_mode()
            elif self.mode == "goto":
                land, summary = self.run_goto_mode()
            
        except KeyboardInterrupt:
            logger.log_warning('TelloWaypointNavCoordinator', 'Application interrupted by user.')
            self.is_running = False
            self.is_mapping_mode = False
            self.is_navigation_mode = False
            self._stop_battery_monitoring()  # Safe cleanup - only stops if running
            TelloWaypointNavCoordinator._active_instance = None
            land = True  # Ensure we land on exit
        except Exception as e:
            logger.log_error('TelloWaypointNavCoordinator', f'Application error: {e}')
            self.is_running = False
            self.is_mapping_mode = False
            self.is_navigation_mode = False
            self._stop_battery_monitoring()  # Safe cleanup - only stops if running
            TelloWaypointNavCoordinator._active_instance = None
            land = True  # Ensure we land on error
        finally:
            if land: 
                self.cleanup()

            return summary

    def _run_mapping_mode(self) -> list:
        """Run the application in mapping mode."""
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
        
        # Initialize drone controller
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
        """Run the application in navigation mode."""
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
        
        # Initialize navigation interface
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
        Run the application in 'goto' mode to navigate to a specific waypoint.
        
        Returns:
            tuple: (land_flag, waypoint_list)
        """
        logger.log_info('TelloWaypointNavCoordinator', 'GOTO MODE ACTIVATED')
        
        try:
            # Check for emergency shutdown at the start
            if TelloWaypointNavCoordinator._emergency_shutdown:
                logger.log_warning('TelloWaypointNavCoordinator', 'Emergency shutdown detected - aborting goto mode')
                return True, [self.current_waypoint]
            
            if not self.is_connected: 
                if not self.connect_drone():
                    logger.log_error('TelloWaypointNavCoordinator', 'Failed to connect to drone. Exiting...')
                    TelloWaypointNavCoordinator._active_instance = None
                    return True, [self.current_waypoint]
            
            if not self.is_flying:
                if not self.takeoff():
                    logger.log_error('TelloWaypointNavCoordinator', 'Failed to take off. Exiting...')
                    TelloWaypointNavCoordinator._active_instance = None
                    return True, [self.current_waypoint]
            
            # Start battery monitoring if not already running
            if not TelloWaypointNavCoordinator._battery_thread_running:
                self._start_battery_monitoring()

            self.is_goto_mode = True
            self.is_running = True
            
            # Check for emergency shutdown after battery monitoring start
            if TelloWaypointNavCoordinator._emergency_shutdown:
                logger.log_warning('TelloWaypointNavCoordinator', 'Emergency shutdown detected during setup')
                return True, [self.current_waypoint]
            
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
                        self._stop_battery_monitoring()
                        TelloWaypointNavCoordinator._active_instance = None
                        return True, [self.current_waypoint]
                    
                    selected_file = waypoint_files[0]  # Latest file
                    logger.log_info('TelloWaypointNavCoordinator', f'Using latest waypoint file: {selected_file}')

                if not self.nav_manager.load_waypoint_file(selected_file):
                    logger.log_error('TelloWaypointNavCoordinator', f'Failed to load waypoint file: {selected_file}')
                    self._stop_battery_monitoring()
                    TelloWaypointNavCoordinator._active_instance = None
                    return True, [self.current_waypoint]
                
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
                if self.instruction == NavigationInstruction.HALT:
                    logger.log_info('TelloWaypointNavCoordinator', f'Stopping at current waypoint "{self.current_waypoint}"')
                    self._stop_battery_monitoring()
                    TelloWaypointNavCoordinator._active_instance = None
                    return True, [self.current_waypoint]
                else:
                    logger.log_info('TelloWaypointNavCoordinator', f'Still at current waypoint "{self.current_waypoint}"')
                    # Keep battery monitoring running
                    TelloWaypointNavCoordinator._active_instance = self
                    return False, [self.current_waypoint]
            
            logger.log_info('TelloWaypointNavCoordinator', f'Navigating to waypoint: {self.waypoint_dest}')
            # Check for emergency shutdown before navigation
            if TelloWaypointNavCoordinator._emergency_shutdown:
                logger.log_warning('TelloWaypointNavCoordinator', 'Emergency shutdown detected before navigation')
                return True, [self.current_waypoint]
            
            success = self.nav_manager.navigate_to_waypoint(self.waypoint_dest, self.tello)
        
            if success:
                # Update current waypoint after navigation
                self.current_waypoint = self.waypoint_dest
                logger.log_success('TelloWaypointNavCoordinator', f'Reached waypoint "{self.current_waypoint}"')

                if self.instruction == NavigationInstruction.HALT:
                    logger.log_info('TelloWaypointNavCoordinator', f'Stopping at waypoint "{self.current_waypoint}"')
                    self._stop_battery_monitoring()
                    TelloWaypointNavCoordinator._active_instance = None
                    return True, [self.current_waypoint]
                else:
                    if TelloWaypointNavCoordinator._emergency_shutdown:
                        logger.log_warning('TelloWaypointNavCoordinator', 'Emergency shutdown detected after navigation')
                        return True, [self.current_waypoint]
                    
                    logger.log_info('TelloWaypointNavCoordinator', f'Continuing at waypoint "{self.current_waypoint}"')
                    # Keep battery monitoring running
                    TelloWaypointNavCoordinator._active_instance = self
                    return False, [self.current_waypoint]
            else:
                logger.log_error('TelloWaypointNavCoordinator', f'Failed to reach waypoint "{self.waypoint_dest}"')
                self._stop_battery_monitoring()
                TelloWaypointNavCoordinator._active_instance = None
                return True, [self.current_waypoint]
                
        except Exception as e:
            logger.log_error('TelloWaypointNavCoordinator', f'Error in goto mode: {e}')
            self._stop_battery_monitoring()
            TelloWaypointNavCoordinator._active_instance = None
            return True, [self.current_waypoint]
        finally: 
            self.is_goto_mode = False
            self.is_running = False
            
    def _find_waypoint_files(self) -> list:
        """Find all available waypoint JSON files."""
        pattern = os.path.join(self.waypoint_dir, "drone_movements_*.json")
        files = glob.glob(pattern)
        return sorted(files, reverse=True)  # Newest first
        
    def display_controls(self):
        """Display control instructions."""
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
        """Connect to the Tello drone."""
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
        """Take off the drone and mark starting waypoint."""
        if not self.is_connected:
            logger.log_error('TelloWaypointNavCoordinator', 'Drone not connected!')
            return False
            
        try:
            logger.log_info('TelloWaypointNavCoordinator', 'Taking off...')
            self.tello.takeoff()
            self.is_flying = True
            time.sleep(2)  # Wait for stabilization
            logger.log_success('TelloWaypointNavCoordinator', 'Drone is airborne!')
            return True
        
        except Exception as e:
            logger.log_error('TelloWaypointNavCoordinator', f'Takeoff failed: {e}')
            return False
    
    def land(self):
        """Land the drone safely."""
        if self.is_flying:
            try:
                logger.log_info('TelloWaypointNavCoordinator', 'Landing drone...')
                self.tello.land()
                self.is_flying = False
                logger.log_success('TelloWaypointNavCoordinator', 'Drone landed successfully!')
            except Exception as e:
                logger.log_error('TelloWaypointNavCoordinator', f'Landing failed: {e}')
    
    def cleanup(self):
        """Cleanup resources and land drone."""
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

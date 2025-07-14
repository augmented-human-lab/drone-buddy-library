from djitellopy import Tello
import time

from .realtime_drone_control import RealTimeDroneController
from .navigation_interface import NavigationInterface

class TelloWaypointNavCoordinator: 
    """
    Class for handling Tello waypoint mapping on Linux.
    """

    _active_instance = None # Class-level instance tracker

    @classmethod
    def get_instance(cls, waypoint_dir: str, vertical_factor: float, movement_speed: int, rotation_speed: int, navigation_speed: int, mode: str, waypoint_dest: str = None, instruction: str = None, create_new: bool = False):
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
            instruction (str, optional): Instruction for goto mode.
            create_new (bool): Whether to create a new instance or return the existing one.
        
        Returns:
            TelloWaypointNavCoordinator instance.
        """
        if create_new: 
            instance = cls(waypoint_dir, vertical_factor, movement_speed, rotation_speed, navigation_speed, mode, waypoint_dest, instruction)
            cls._active_instance = instance
            return instance
        else: 
            instance = cls._active_instance
            instance.waypoint_dest = waypoint_dest
            instance.instruction = instruction
            return instance 

    def __init__(self, waypoint_dir: str, vertical_factor: float, movement_speed: int, rotation_speed: int, navigation_speed: int, mode: str, waypoint_dest: str = None, instruction: str = None):
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
        # Initialize parameters
        self.waypoint_dir = waypoint_dir
        self.vertical_factor = vertical_factor
        self.movement_speed = movement_speed
        self.rotation_speed = rotation_speed
        self.navigation_speed = navigation_speed
        self.mode = mode
        self.waypoint_dest = waypoint_dest
        self.instruction = instruction

        # Initialize Tello drone
        self.tello = Tello()

        # Initialize application state
        self.is_connected = False
        self.is_flying = False
        self.is_mapping_mode = False
        self.is_navigation_mode = False
        self.is_running = False
        self.current_waypoint = "WP_001"
    
    def run(self):
        """Run the main application."""
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
            print("\n🛑 Application interrupted by user")
            self.is_running = False
            self.is_mapping_mode = False
            self.is_navigation_mode = False
            TelloWaypointNavCoordinator._active_instance = None
            land = True  # Ensure we land on exit
        except Exception as e:
            print(f"\n❌ Application error: {e}")
            self.is_running = False
            self.is_mapping_mode = False
            self.is_navigation_mode = False
            TelloWaypointNavCoordinator._active_instance = None
            land = True  # Ensure we land on error
        finally:
            if land: 
                self._cleanup()

            return summary

    def _run_mapping_mode(self) -> list:
        """Run the application in mapping mode."""
        print("\n🗺️  MAPPING MODE ACTIVATED")
        print("You will create waypoints by manually controlling the drone.")

        if self.is_connected or self.is_flying:
            print("Drone is already connected or flying. Please land it first.")
            return []

        self.display_controls()

        if not self.connect_drone():
            print("Failed to connect to drone. Exiting...")
            return []
        
        if not self.takeoff():
            print("Failed to take off. Exiting...")
            return []
        
        # Initialize drone controller
        self.drone_controller = RealTimeDroneController(self.waypoint_dir, self.movement_speed, self.rotation_speed)
        
        self.is_mapping_mode = True
        self.is_running = True
    
        print("Use keyboard controls to move drone and create waypoints.")
        
        # Start user interface
        try:
            summary = self.drone_controller.run(drone_instance=self.tello)
        except Exception as e:
            print(f"Error during execution: {e}")
            summary = []
        finally:
            self.is_mapping_mode = False
            self.is_running = False

            return summary  # Return summary of waypoints created
    
    def _run_navigation_mode(self) -> list:
        """Run the application in navigation mode."""
        print("\n🧭 NAVIGATION MODE ACTIVATED")
        
        history = []

        if self.is_connected or self.is_flying:
            print("Drone is already connected or flying. Please land it first.")
            return []
        
         # Connect and takeoff
        if not self.connect_drone():
            print("Failed to connect to drone. Exiting...")
            return []
        
        if not self.takeoff():
            print("Failed to take off. Exiting...")
            return []
        
        # Initialize navigation interface
        self.nav_interface = NavigationInterface(self.waypoint_dir, self.vertical_factor, self.navigation_speed)
        
        self.is_navigation_mode = True
        self.is_running = True
        
        try: 
            history = self.nav_interface.run(drone_instance=self.tello)
        except Exception as e:
            print(f"Error during navigation: {e}")
        finally: 
            self.is_navigation_mode = False
            self.is_running = False
            return history  # Return navigation history
    
    def run_goto_mode(self): 
        """
        Run the application in 'goto' mode to navigate to a specific waypoint.
        
        Returns:
            list: A list of waypoints navigated to.
        """
        print("\n🚀 GOTO MODE ACTIVATED")
        
        if not self.is_connected: 
            if not self.connect_drone():
                print("Failed to connect to drone. Exiting...")
                TelloWaypointNavCoordinator._active_instance = None
                return True, [self.current_waypoint]
        
        if not self.is_flying:
            if not self.takeoff():
                print("Failed to take off. Exiting...")
                TelloWaypointNavCoordinator._active_instance = None
                return True, [self.current_waypoint]
        
        if not hasattr(self, 'nav_manager'):
            from .waypoint_navigation import WaypointNavigationManager
            self.nav_manager = WaypointNavigationManager(nav_speed=self.navigation_speed, vertical_factor=self.vertical_factor)

            waypoint_files = self._find_waypoint_files()
            if not waypoint_files:
                print("❌ No waypoint files found. Please run mapping mode first.")
                TelloWaypointNavCoordinator._active_instance = None
                return True, [self.current_waypoint]

            latest_file = waypoint_files[0]
            if not self.nav_manager.load_waypoint_file(latest_file):
                print(f"❌ Failed to load waypoint file: {latest_file}")
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
            print(f"❌ Waypoint '{self.waypoint_dest}' not found")
            if self.instruction.lower() == "halt":
                print(f"Stopping at current waypoint '{self.current_waypoint}'")
                TelloWaypointNavCoordinator._active_instance = None
                return True, [self.current_waypoint]
            else:
                print(f"Still at current waypoint '{self.current_waypoint}'")
                TelloWaypointNavCoordinator._active_instance = self
                return False, [self.current_waypoint]
        
        print(f"Navigating to waypoint: {self.waypoint_dest}")
        success = self.nav_manager.navigate_to_waypoint(self.waypoint_dest, self.tello)
    
        if success:
            # Update current waypoint after navigation
            self.current_waypoint = self.waypoint_dest
            print(f"✅ Reached waypoint '{self.current_waypoint}'")

            if self.instruction.lower() == "halt":
                print(f"Stopping at waypoint '{self.current_waypoint}'")
                TelloWaypointNavCoordinator._active_instance = None
                return True, [self.current_waypoint]
            else:
                TelloWaypointNavCoordinator._active_instance = self
                return False, [self.current_waypoint]
        else:
            print(f"❌ Failed to reach waypoint '{self.waypoint_dest}'")
            TelloWaypointNavCoordinator._active_instance = None
            return True, [self.current_waypoint]
            
    def _find_waypoint_files(self) -> list:
        """Find all available waypoint JSON files."""
        import glob
        import os
        pattern = os.path.join(self.waypoint_dir, "drone_movements_*.json")
        files = glob.glob(pattern)
        return sorted(files, reverse=True)  # Newest first
        
    def display_controls(self):
        """Display control instructions."""
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
        print("\nNOTES:")
        print("- Hold key to move, release to stop")
        print("- Only one movement/action at a time")
        print("- All movements are recorded automatically")
        print("="*50)
        print()

    def connect_drone(self):
        """Connect to the Tello drone."""
        try:
            print("Connecting to Tello drone...")
            self.tello.RESPONSE_TIMEOUT = 7
            self.tello.connect(wait_for_state=False)
            print("Drone connected successfully!")

            self.is_connected = True

            try:
                battery_response = self.tello.send_command_with_return("battery?", timeout=5)
                print(f"✅ Battery: {battery_response}%")
            except Exception as e:
                print(f"❌ Battery command failed: {e}")
            
            return True
        except Exception as e:
            print(f"Failed to connect to drone: {e}")
            return False
    
    def takeoff(self):
        """Take off the drone and mark starting waypoint."""
        if not self.is_connected:
            print("Drone not connected!")
            return False
            
        try:
            print("Taking off...")
            self.tello.takeoff()
            self.is_flying = True
            time.sleep(2)  # Wait for stabilization
            print("Drone is airborne! 🛫")
            return True
        
        except Exception as e:
            print(f"Takeoff failed: {e}")
            return False
    
    def land(self):
        """Land the drone safely."""
        if self.is_flying:
            try:
                print("Landing drone...")
                self.tello.land()
                self.is_flying = False
                print("Drone landed successfully!")
            except Exception as e:
                print(f"Landing failed: {e}")
    
    def _cleanup(self):
        """Cleanup resources and land drone."""
        print("\n🧹 Cleaning up...")

        if self.is_flying:
            try: 
                print("Landing drone...")
                self.tello.land()
                self.is_flying = False
            except Exception as e:
                print(f"Error during landing: {e}")
        
        if self.is_connected:
            try:
                print("Disconnecting from drone...")
                self.tello.end()
                self.is_connected = False
            except Exception as e:
                print(f"Error during disconnection: {e}")

        print("👋 Application closed successfully")

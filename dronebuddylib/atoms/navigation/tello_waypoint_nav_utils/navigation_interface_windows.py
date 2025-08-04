#!/usr/bin/env python3
"""
Windows-specific navigation interface for drone waypoint navigation.

Provides user interface for navigating between recorded waypoints using Windows-compatible
input handling (msvcrt). Manages waypoint file selection, battery monitoring,
and interactive navigation menus.
"""
import os
import glob
import time
from typing import Optional

import msvcrt
from typing import Optional
from .waypoint_navigation import WaypointNavigationManager

from dronebuddylib.utils.logger import Logger

logger = Logger()

class NavigationInterfaceWindows:
    """Windows navigation interface for waypoint-based drone navigation."""
    
    def __init__(self, waypoint_dir: str, vertical_factor: float, nav_speed: int, waypoint_file: str = None):
        """Initialize navigation interface with speed and directory settings."""
        logger.log_info('NavigationInterfaceWindows', 'Initializing navigation interface.')
        self.nav_manager = WaypointNavigationManager(nav_speed=nav_speed, vertical_factor=vertical_factor)
        self.is_running = True
        self.waypoint_dir = waypoint_dir
        self.waypoint_file = waypoint_file
        self._input_ready = False  # Windows input state tracking
        self._user_input = ""      # Windows input buffer
        logger.log_debug('NavigationInterfaceWindows', f'Initialized with waypoint_dir={waypoint_dir}, vertical_factor={vertical_factor}, nav_speed={nav_speed}')
    
    def _wait_for_input_with_timeout(self, timeout_seconds: int) -> Optional[str]:
        """Windows-compatible input with timeout using msvcrt for character-by-character input."""
        start_time = time.time()
        input_buffer = ""
        
        while time.time() - start_time < timeout_seconds:
            if msvcrt.kbhit():  # Check if key was pressed
                char = msvcrt.getch()
                if char == b'\r':  # Enter key - submit input
                    print()  # New line after input
                    return input_buffer.strip()
                elif char == b'\x08':  # Backspace - remove character
                    if input_buffer:
                        input_buffer = input_buffer[:-1]
                        print('\b \b', end='', flush=True)  # Visual backspace
                elif char == b'\x03':  # Ctrl+C - interrupt
                    raise KeyboardInterrupt
                else:
                    try:
                        decoded_char = char.decode('utf-8')
                        input_buffer += decoded_char
                        print(decoded_char, end='', flush=True)  # Echo character
                    except UnicodeDecodeError:
                        continue  # Skip invalid characters
            time.sleep(0.1)  # Prevent high CPU usage during polling
        
        return None  # Timeout occurred
    
    def run(self, drone_instance=None) -> list:
        """Main entry point - loads waypoints and starts navigation session."""
        logger.log_info('NavigationInterfaceWindows', 'Starting navigation interface.')
        history = []
        try:
            if drone_instance is None:
                logger.log_error('NavigationInterfaceWindows', 'No drone instance provided.')
                return []  # Return empty list instead of None
            
            print("\n🧭 WAYPOINT NAVIGATION SYSTEM")
            print("=" * 50)
            
            # Load waypoint file (auto-select if single file, user selection if multiple)
            if not self._load_waypoint_file(drone_instance=drone_instance):
                return []  # Return empty list if file loading fails
            
            # Enter interactive navigation loop
            history = self._navigation_loop(drone_instance=drone_instance)
            
        except KeyboardInterrupt:
            logger.log_warning('NavigationInterfaceWindows', 'Navigation interrupted by user.')
        except Exception as e:
            logger.log_error('NavigationInterfaceWindows', f'Navigation error: {e}')
        finally:
            drone_instance.send_rc_control(0, 0, 0, 0)  # Emergency stop
            logger.log_info('NavigationInterfaceWindows', 'Navigation system closed.')
            return history
    
    def _load_waypoint_file(self, drone_instance=None) -> bool:
        """Load waypoint file - uses specified file or prompts user to select from available files."""
        logger.log_info('NavigationInterfaceWindows', 'Loading waypoint file.')
        
        # Check if specific waypoint file was provided
        if self.waypoint_file is not None:
            specified_file_path = os.path.join(self.waypoint_dir, self.waypoint_file)
            
            if os.path.exists(specified_file_path):
                logger.log_info('NavigationInterfaceWindows', f'Found specified waypoint file: {specified_file_path}')
                return self.nav_manager.load_waypoint_file(specified_file_path)
            else:
                logger.log_warning('NavigationInterfaceWindows', f'Specified waypoint file not found: {specified_file_path}, proceeding with file selection.')
                self.waypoint_file = None  # Clear invalid file and fallback to user selection
        
        # Find all available waypoint files
        waypoint_files = self._find_waypoint_files()
        
        if not waypoint_files:
            logger.log_error('NavigationInterfaceWindows', 'No waypoint files found.')
            return False
        
        if len(waypoint_files) == 1:
            # Auto-load single file
            selected_file = waypoint_files[0]
            logger.log_info('NavigationInterfaceWindows', f'Found single waypoint file: {selected_file}')
        else:
            # Let user choose from multiple files
            logger.log_info('NavigationInterfaceWindows', f'Found {len(waypoint_files)} waypoint files, requesting user selection.')
            selected_file = self._select_waypoint_file(waypoint_files, drone_instance=drone_instance)
            if not selected_file:
                return False
        
        return self.nav_manager.load_waypoint_file(selected_file)
    
    def _find_waypoint_files(self) -> list:
        """Find all waypoint JSON files in directory, sorted newest first."""
        pattern = os.path.join(self.waypoint_dir, "drone_movements_*.json")
        files = glob.glob(pattern)
        return sorted(files, reverse=True)  # Newest first
    
    def _select_waypoint_file(self, files: list, drone_instance=None) -> Optional[str]:
        """Display file selection menu and get user choice with battery monitoring."""
        print(f"\n📁 Found {len(files)} waypoint files:")
        print("-" * 50)
        
        for i, file in enumerate(files, 1):
            # Extract timestamp from filename for user-friendly display
            filename = os.path.basename(file)
            timestamp = filename.replace('drone_movements_', '').replace('.json', '')
            print(f"  {i}. {filename} (Created: {timestamp})")
        
        while True:
            try:
                # Check battery level during file selection
                try:
                    battery_str = drone_instance.send_command_with_return("battery?", timeout=5)
                    battery = int(battery_str)
                    if battery < 20:
                        logger.log_warning('NavigationInterfaceWindows', f'Low battery detected: {battery}%')
                        if battery < 10:
                            logger.log_error('NavigationInterfaceWindows', f'CRITICAL: Battery too low ({battery}%), landing...')
                            return None
                except Exception as e:
                    logger.log_error('NavigationInterfaceWindows', f'Error checking battery: {e}')
                    return None
                
                prompt = f"\nSelect waypoint file (1-{len(files)}) or 'q' to quit: "
                print(prompt, end='', flush=True)

                # Windows: Use msvcrt for 5-second timeout
                choice = self._wait_for_input_with_timeout(5)

                if choice is not None:
                    choice = choice.lower()
                    if choice == 'q':
                        return None
                    
                    try: 
                        file_index = int(choice) - 1
                        if 0 <= file_index < len(files):
                            return files[file_index]
                        else:
                            print(f"❌ Invalid choice. Please enter 1-{len(files)}")
                    except ValueError:
                        print("❌ Invalid input. Please enter a valid number.")
                else:
                    # Timeout occurred - clear line and continue
                    print("\r" + " " * 50 + "\r", end='')
                    continue
                    
            except Exception as e:
                logger.log_error('NavigationInterfaceWindows', f'Error reading input: {e}')
                return None
    
    def _navigation_loop(self, drone_instance=None):
        """Interactive loop for waypoint selection and navigation execution."""
        loop_count = 0
        waypoints_history = []
        try:
            while self.is_running:
                # Display current position and available destinations
                destinations = self.nav_manager.print_navigation_options()
                
                if not destinations:
                    logger.log_info('NavigationInterfaceWindows', 'No other waypoints to navigate to.')
                    break
                
                # Get user's navigation choice
                choice = self._get_navigation_choice(destinations, loop_count, drone_instance=drone_instance)
                
                if choice == 'quit':
                    logger.log_info('NavigationInterfaceWindows', 'User chose to quit navigation.')
                    break
                elif choice == 'reload':
                    logger.log_info('NavigationInterfaceWindows', 'User chose to reload waypoint file.')
                    if self._load_waypoint_file(drone_instance=drone_instance):
                        continue
                    else:
                        break
                elif isinstance(choice, str):
                    # Execute navigation to selected waypoint (waypoint ID returned)
                    logger.log_info('NavigationInterfaceWindows', f'User selected waypoint: {choice}')
                    success = self.nav_manager.navigate_to_waypoint(choice, drone_instance=drone_instance)
                    if success:
                        logger.log_success('NavigationInterfaceWindows', f'Navigation to {choice} completed!')
                        waypoints_history.append(choice)
                        loop_count += 1
                    else:
                        logger.log_error('NavigationInterfaceWindows', f'Navigation to {choice} failed!')
                        break
                
        except Exception as e:
            logger.log_error('NavigationInterfaceWindows', f'Error in navigation loop: {e}')
        finally: 
            return waypoints_history
    
    def _get_navigation_choice(self, destinations: list, loopCount: int, drone_instance=None) -> str:
        """Display navigation menu and get user's waypoint choice with battery monitoring."""
        print(f"\n🎮 NAVIGATION OPTIONS:")
        print("-" * 30)
        
        for i, (wp_id, wp_name) in enumerate(destinations, 1):
            print(f"  {i}. Navigate to '{wp_name}' ({wp_id})")
        
        print(f"  r. Reload waypoint file")
        print(f"  q. Quit navigation")
        
        while True:
            try:
                # Monitor battery level during menu
                try:
                    battery_str = drone_instance.send_command_with_return("battery?", timeout=5)
                    battery = int(battery_str)
                    if battery < 20:
                        logger.log_warning('NavigationInterfaceWindows', f'Low battery detected: {battery}%')
                        if battery < 10:
                            logger.log_error('NavigationInterfaceWindows', f'CRITICAL: Battery too low ({battery}%), landing...')
                            return 'quit'
                except Exception as e:
                    logger.log_error('NavigationInterfaceWindows', f'Error checking battery: {e}')
                    return 'quit'

                # Show different prompt based on loop count (reload only available at start)
                if loopCount == 0:
                    prompt = f"\nEnter your choice (1-{len(destinations)}, r, q): "
                else: 
                    prompt = f"\nEnter your choice (1-{len(destinations)}, q): "

                print(prompt, end='', flush=True)

                # Windows: Use msvcrt for 5-second timeout
                choice = self._wait_for_input_with_timeout(5)

                if choice is not None:
                    choice = choice.lower()
                    if choice == 'q':
                        return 'quit'
                    elif choice == 'r':
                        if loopCount == 0: 
                            print("❗ Reloading waypoint file...")
                            return 'reload'
                        else: 
                            print("❗ You can only reload the waypoint file at the start of navigation.")
                            continue
                    else:
                        # Parse waypoint selection
                        try:
                            waypoint_index = int(choice) - 1
                            if 0 <= waypoint_index < len(destinations):
                                return destinations[waypoint_index][0]  # Return waypoint ID
                            else:
                                print(f"❌ Invalid choice. Please enter 1-{len(destinations)}")
                        except ValueError:
                            print("❌ Invalid input. Please enter a valid number.")
                
                else: 
                    # Timeout occurred - clear line and continue
                    print("\r" + " " * 50 + "\r", end='')
                    continue
                        
            except KeyboardInterrupt:
                return 'quit'
            except Exception as e:
                logger.log_error('NavigationInterfaceWindows', f'Error reading input: {e}')
                return 'quit'

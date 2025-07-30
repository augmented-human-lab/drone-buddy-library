#!/usr/bin/env python3
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
    """User interface for waypoint navigation on Windows."""
    
    def __init__(self, waypoint_dir: str, vertical_factor: float, nav_speed: int, waypoint_file: str = None):
        logger.log_info('NavigationInterfaceWindows', 'Initializing navigation interface.')
        self.nav_manager = WaypointNavigationManager(nav_speed=nav_speed, vertical_factor=vertical_factor)
        self.is_running = True
        self.waypoint_dir = waypoint_dir
        self.waypoint_file = waypoint_file
        self._input_ready = False
        self._user_input = ""
        logger.log_debug('NavigationInterfaceWindows', f'Initialized with waypoint_dir={waypoint_dir}, vertical_factor={vertical_factor}, nav_speed={nav_speed}')
    
    def _wait_for_input_with_timeout(self, timeout_seconds: int) -> Optional[str]:
        """Windows-compatible input with timeout using msvcrt."""
        start_time = time.time()
        input_buffer = ""
        
        while time.time() - start_time < timeout_seconds:
            if msvcrt.kbhit():
                char = msvcrt.getch()
                if char == b'\r':  # Enter key
                    print()  # New line after input
                    return input_buffer.strip()
                elif char == b'\x08':  # Backspace
                    if input_buffer:
                        input_buffer = input_buffer[:-1]
                        print('\b \b', end='', flush=True)
                elif char == b'\x03':  # Ctrl+C
                    raise KeyboardInterrupt
                else:
                    try:
                        decoded_char = char.decode('utf-8')
                        input_buffer += decoded_char
                        print(decoded_char, end='', flush=True)
                    except UnicodeDecodeError:
                        continue
            time.sleep(0.1)  # Small delay to prevent high CPU usage
        
        return None  # Timeout
    
    def run(self, drone_instance=None) -> list:
        """Run the navigation interface."""
        logger.log_info('NavigationInterfaceWindows', 'Starting navigation interface.')
        history = []
        try:
            if drone_instance is None:
                logger.log_error('NavigationInterfaceWindows', 'No drone instance provided.')
                return
            
            print("\n🧭 WAYPOINT NAVIGATION SYSTEM")
            print("=" * 50)
            
            # Load waypoint file
            if not self._load_waypoint_file(drone_instance=drone_instance):
                return
            
            # Main navigation loop
            history = self._navigation_loop(drone_instance=drone_instance)
            
        except KeyboardInterrupt:
            logger.log_warning('NavigationInterfaceWindows', 'Navigation interrupted by user.')
        except Exception as e:
            logger.log_error('NavigationInterfaceWindows', f'Navigation error: {e}')
        finally:
            drone_instance.send_rc_control(0, 0, 0, 0)  # Stop any ongoing movement
            logger.log_info('NavigationInterfaceWindows', 'Navigation system closed.')
            return history
    
    def _load_waypoint_file(self, drone_instance=None) -> bool:
        """Load waypoint file with user selection."""
        logger.log_info('NavigationInterfaceWindows', 'Loading waypoint file.')
        
        # Check if specific waypoint_file is specified
        if self.waypoint_file is not None:
            # Construct the full path to the specified waypoint file
            specified_file_path = os.path.join(self.waypoint_dir, self.waypoint_file)
            
            # Check if the specified file exists
            if os.path.exists(specified_file_path):
                logger.log_info('NavigationInterfaceWindows', f'Found specified waypoint file: {specified_file_path}')
                return self.nav_manager.load_waypoint_file(specified_file_path)
            else:
                logger.log_warning('NavigationInterfaceWindows', f'Specified waypoint file not found: {specified_file_path}, proceeding with file selection.')
                self.waypoint_file = None  # Reset
                # Fall through to normal file selection process
        
        # Find available waypoint files
        waypoint_files = self._find_waypoint_files()
        
        if not waypoint_files:
            logger.log_error('NavigationInterfaceWindows', 'No waypoint files found.')
            return False
        
        if len(waypoint_files) == 1:
            # Only one file, load it automatically
            selected_file = waypoint_files[0]
            logger.log_info('NavigationInterfaceWindows', f'Found single waypoint file: {selected_file}')
        else:
            # Multiple files, let user choose
            logger.log_info('NavigationInterfaceWindows', f'Found {len(waypoint_files)} waypoint files, requesting user selection.')
            selected_file = self._select_waypoint_file(waypoint_files, drone_instance=drone_instance)
            if not selected_file:
                return False
        
        return self.nav_manager.load_waypoint_file(selected_file)
    
    def _find_waypoint_files(self) -> list:
        """Find all available waypoint JSON files."""
        pattern = os.path.join(self.waypoint_dir, "drone_movements_*.json")
        files = glob.glob(pattern)
        return sorted(files, reverse=True)  # Newest first
    
    def _select_waypoint_file(self, files: list, drone_instance=None) -> Optional[str]:
        """Let user select which waypoint file to use."""
        print(f"\n📁 Found {len(files)} waypoint files:")
        print("-" * 50)
        
        for i, file in enumerate(files, 1):
            # Extract timestamp from filename for better display
            timestamp = file.replace('drone_movements_', '').replace('.json', '')
            print(f"  {i}. {file} (Created: {timestamp})")
        
        while True:
            try:
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

                # Wait for input with 5-second timeout using Windows-compatible method
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
                        print("❌ Invalid input. Please enter a valid option.")
                else:
                    # No input received within timeout
                    print("\r" + " " * 50 + "\r", end='')
                    continue
                    
            except Exception as e:
                logger.log_error('NavigationInterfaceWindows', f'Error reading input: {e}')
                return None
    
    def _navigation_loop(self, drone_instance=None):
        """Main navigation interaction loop."""
        loop_count = 0
        waypoints_history = []
        try:
            while self.is_running:
                # Show current position and options
                destinations = self.nav_manager.print_navigation_options()
                
                if not destinations:
                    logger.log_info('NavigationInterfaceWindows', 'No other waypoints to navigate to.')
                    break
                
                # Get user choice
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
                    # Navigate to selected waypoint
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
        """Get navigation choice from user."""
        print(f"\n🎮 NAVIGATION OPTIONS:")
        print("-" * 30)
        
        for i, (wp_id, wp_name) in enumerate(destinations, 1):
            print(f"  {i}. Navigate to '{wp_name}' ({wp_id})")
        
        print(f"  r. Reload waypoint file")
        print(f"  q. Quit navigation")
        
        while True:
            try:
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

                if loopCount == 0:
                    prompt = f"\nEnter your choice (1-{len(destinations)}, r, q): "
                else: 
                    prompt = f"\nEnter your choice (1-{len(destinations)}, q): "

                print(prompt, end='', flush=True)

                # Wait for input with 5-second timeout using Windows-compatible method
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
                        # Try to parse as waypoint selection
                        try:
                            waypoint_index = int(choice) - 1
                            if 0 <= waypoint_index < len(destinations):
                                return destinations[waypoint_index][0]  # Return waypoint ID
                            else:
                                print(f"❌ Invalid choice. Please enter 1-{len(destinations)}")
                        except ValueError:
                            print("❌ Invalid input. Please enter a valid option.")
                
                else: 
                    # No input received within timeout
                    print("\r" + " " * 50 + "\r", end='')
                    continue
                        
            except KeyboardInterrupt:
                return 'quit'
            except Exception as e:
                logger.log_error('NavigationInterfaceWindows', f'Error reading input: {e}')
                return 'quit'

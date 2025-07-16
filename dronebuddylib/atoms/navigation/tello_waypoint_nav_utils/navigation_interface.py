#!/usr/bin/env python3
import os
import glob
import sys
from typing import Optional

import select
from typing import Optional
from .waypoint_navigation import WaypointNavigationManager

from dronebuddylib.utils.logger import Logger

logger = Logger()

class NavigationInterface:
    """User interface for waypoint navigation."""
    
    def __init__(self, waypoint_dir: str, vertical_factor: float, nav_speed: int):
        logger.log_info('NavigationInterface', 'Initializing navigation interface.')
        self.nav_manager = WaypointNavigationManager(nav_speed=nav_speed, vertical_factor=vertical_factor)
        self.is_running = True
        self.waypoint_dir = waypoint_dir
        logger.log_debug('NavigationInterface', f'Initialized with waypoint_dir={waypoint_dir}, vertical_factor={vertical_factor}, nav_speed={nav_speed}')
    
    def run(self, drone_instance=None) -> list:
        """Run the navigation interface."""
        logger.log_info('NavigationInterface', 'Starting navigation interface.')
        history = []
        try:
            if drone_instance is None:
                logger.log_error('NavigationInterface', 'No drone instance provided.')
                return
            
            print("\n🧭 WAYPOINT NAVIGATION SYSTEM")
            print("=" * 50)
            
            # Load waypoint file
            if not self._load_waypoint_file(drone_instance=drone_instance):
                return
            
            # Main navigation loop
            history = self._navigation_loop(drone_instance=drone_instance)
            
        except KeyboardInterrupt:
            logger.log_warning('NavigationInterface', 'Navigation interrupted by user.')
        except Exception as e:
            logger.log_error('NavigationInterface', f'Navigation error: {e}')
        finally:
            drone_instance.send_rc_control(0, 0, 0, 0)  # Stop any ongoing movement
            logger.log_info('NavigationInterface', 'Navigation system closed.')
            return history
    
    def _load_waypoint_file(self, drone_instance=None) -> bool:
        """Load waypoint file with user selection."""
        logger.log_info('NavigationInterface', 'Loading waypoint file.')
        
        # Find available waypoint files
        waypoint_files = self._find_waypoint_files()
        
        if not waypoint_files:
            logger.log_error('NavigationInterface', 'No waypoint files found.')
            return False
        
        if len(waypoint_files) == 1:
            # Only one file, load it automatically
            selected_file = waypoint_files[0]
            logger.log_info('NavigationInterface', f'Found single waypoint file: {selected_file}')
        else:
            # Multiple files, let user choose
            logger.log_info('NavigationInterface', f'Found {len(waypoint_files)} waypoint files, requesting user selection.')
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
                        logger.log_warning('NavigationInterface', f'Low battery detected: {battery}%')
                        if battery < 10:
                            logger.log_error('NavigationInterface', f'CRITICAL: Battery too low ({battery}%), landing...')
                            return None
                except Exception as e:
                    logger.log_error('NavigationInterface', f'Error checking battery: {e}')
                    return None
                
                prompt = f"\nSelect waypoint file (1-{len(files)}) or 'q' to quit: "

                print(prompt, end='', flush=True)

                # Wait for input with 5-second timeout
                ready, _, _ = select.select([sys.stdin], [], [], 5)

                if ready:
                    choice = sys.stdin.readline().strip().lower()
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
                logger.log_error('NavigationInterface', f'Error reading input: {e}')
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
                    logger.log_info('NavigationInterface', 'No other waypoints to navigate to.')
                    break
                
                # Get user choice
                choice = self._get_navigation_choice(destinations, loop_count, drone_instance=drone_instance)
                
                if choice == 'quit':
                    logger.log_info('NavigationInterface', 'User chose to quit navigation.')
                    break
                elif choice == 'reload':
                    logger.log_info('NavigationInterface', 'User chose to reload waypoint file.')
                    if self._load_waypoint_file(drone_instance=drone_instance):
                        continue
                    else:
                        break
                elif isinstance(choice, str):
                    # Navigate to selected waypoint
                    logger.log_info('NavigationInterface', f'User selected waypoint: {choice}')
                    success = self.nav_manager.navigate_to_waypoint(choice, drone_instance=drone_instance)
                    if success:
                        logger.log_success('NavigationInterface', f'Navigation to {choice} completed!')
                        waypoints_history.append(choice)
                        loop_count += 1
                    else:
                        logger.log_error('NavigationInterface', f'Navigation to {choice} failed!')
                        break
                
        except Exception as e:
            logger.log_error('NavigationInterface', f'Error in navigation loop: {e}')
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
                        logger.log_warning('NavigationInterface', f'Low battery detected: {battery}%')
                        if battery < 10:
                            logger.log_error('NavigationInterface', f'CRITICAL: Battery too low ({battery}%), landing...')
                            return 'quit'
                except Exception as e:
                    logger.log_error('NavigationInterface', f'Error checking battery: {e}')
                    return 'quit'

                if loopCount == 0:
                    prompt = f"\nEnter your choice (1-{len(destinations)}, r, q): "
                else: 
                    prompt = f"\nEnter your choice (1-{len(destinations)}, q): "

                print(prompt, end='', flush=True)

                # Wait for input with 5-second timeout
                ready, _, _ = select.select([sys.stdin], [], [], 5)

                if ready:
                    choice = sys.stdin.readline().strip().lower()
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
                logger.log_error('NavigationInterface', f'Error reading input: {e}')
                return 'quit'
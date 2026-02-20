#!/usr/bin/env python3
"""
Linux-specific navigation interface for 2D hierarchical drone waypoint navigation.

Provides user interface for navigating between Super Waypoints and Inner Waypoints
using Linux-compatible input handling (select/stdin). Manages waypoint file selection,
battery monitoring, and interactive navigation menus.

Key Features:
- 2D waypoint structure support (Super Waypoints + Inner Waypoints)
- Smart navigation routing through Super Waypoint hubs
- Linux-compatible input handling
- Battery monitoring during navigation
"""
import os
import glob
import sys
import time
import traceback
from typing import Optional

import select
from .waypoint_navigation import WaypointNavigationManager

from dronebuddylib.utils.logger import Logger

logger = Logger()


class NavigationInterface:
    """Linux navigation interface for 2D hierarchical waypoint-based drone navigation."""
    
    def __init__(self, waypoint_dir: str, vertical_factor: float, nav_speed: int, waypoint_file: str = None,
                 obstacle_detector=None, frame_read=None):
        """Initialize navigation interface with speed and directory settings.
        
        Args:
            waypoint_dir: Directory containing waypoint files
            vertical_factor: Vertical movement scaling factor
            nav_speed: Navigation speed in cm/s
            waypoint_file: Optional specific waypoint file to use
            obstacle_detector: Optional MiDaSObstacleDetector for depth-based obstacle detection
            frame_read: Optional drone frame reader for video streaming
        """
        logger.log_info('NavigationInterface', 'Initializing navigation interface.')
        self.nav_manager = WaypointNavigationManager(
            nav_speed=nav_speed, 
            vertical_factor=vertical_factor,
            obstacle_detector=obstacle_detector,
            frame_read=frame_read
        )
        self.is_running = True
        self.waypoint_dir = waypoint_dir
        self.waypoint_file = waypoint_file
        self._input_ready = False
        self._user_input = ""
        self.frame_read = frame_read  # Store for video display
        logger.log_debug('NavigationInterface', 
            f'Initialized with waypoint_dir={waypoint_dir}, vertical_factor={vertical_factor}, nav_speed={nav_speed}, '
            f'obstacle_detection={obstacle_detector is not None}')
    
    def run(self, drone_instance=None) -> list:
        """Main entry point - loads 2D waypoints and starts navigation session."""
        logger.log_info('NavigationInterface', 'Starting navigation interface.')
        history = []
        try:
            if drone_instance is None:
                logger.log_error('NavigationInterface', 'No drone instance provided.')
                return []
            
            print("\n" + "=" * 60)
            print("🧭 2D HIERARCHICAL WAYPOINT NAVIGATION SYSTEM")
            print("=" * 60)
            print("\nThis system navigates through a web of interconnected waypoints:")
            print("- ⭐ Super Waypoints: Hub points that connect different areas")
            print("- 📍 Inner Waypoints: Local points within each Super Waypoint zone")
            print("\nSmart routing automatically finds the best path through hubs!")
            print("=" * 60)
            
            # Load waypoint file (auto-select if single file, user selection if multiple)
            if not self._load_waypoint_file(drone_instance=drone_instance):
                return []
            
            # Enter interactive navigation loop
            history = self._navigation_loop(drone_instance=drone_instance)
            
        except KeyboardInterrupt:
            logger.log_warning('NavigationInterface', 'Navigation interrupted by user.')
        except Exception as e:
            logger.log_error('NavigationInterface', f'Navigation error: {e}')
            traceback.print_exc()
        finally:
            drone_instance.send_rc_control(0, 0, 0, 0)
            logger.log_info('NavigationInterface', 'Navigation system closed.')
            return history
    
    def _load_waypoint_file(self, drone_instance=None) -> bool:
        """Load 2D waypoint file - uses specified file or prompts for selection."""
        logger.log_info('NavigationInterface', 'Loading waypoint file.')
        
        # Check if specific file was provided
        if self.waypoint_file is not None:
            specified_file_path = os.path.join(self.waypoint_dir, self.waypoint_file)
            
            if os.path.exists(specified_file_path):
                logger.log_info('NavigationInterface', 
                    f'Found specified waypoint file: {specified_file_path}')
                return self.nav_manager.load_waypoint_file(specified_file_path)
            else:
                logger.log_warning('NavigationInterface', 
                    f'Specified file not found: {specified_file_path}')
                self.waypoint_file = None
        
        # Find available waypoint files
        waypoint_files = self._find_waypoint_files()
        
        if not waypoint_files:
            logger.log_error('NavigationInterface', 'No waypoint files found.')
            print("\n❌ No waypoint files found in the directory.")
            print("   Please run mapping mode first to create a waypoint file.")
            return False
        
        if len(waypoint_files) == 1:
            selected_file = waypoint_files[0]
            logger.log_info('NavigationInterface', 
                f'Found single waypoint file: {selected_file}')
        else:
            logger.log_info('NavigationInterface', 
                f'Found {len(waypoint_files)} waypoint files')
            selected_file = self._select_waypoint_file(waypoint_files, drone_instance=drone_instance)
            if not selected_file:
                return False
        
        return self.nav_manager.load_waypoint_file(selected_file)
    
    def _find_waypoint_files(self) -> list:
        """Find all waypoint JSON files in directory, sorted newest first."""
        # Look for 2D format files (format version 2.0)
        pattern = os.path.join(self.waypoint_dir, "drone_movements_*.json")
        files = glob.glob(pattern)
        return sorted(files, reverse=True)
    
    def _select_waypoint_file(self, files: list, drone_instance=None) -> Optional[str]:
        """Display file selection menu and get user choice."""
        print(f"\n📁 Found {len(files)} waypoint files:")
        print("-" * 50)
        
        for i, file in enumerate(files, 1):
            filename = os.path.basename(file)
            timestamp = filename.replace('drone_movements_', '').replace('.json', '')
            print(f"  {i}. {filename}")
            print(f"     Created: {timestamp}")
        
        while True:
            try:
                # Battery check
                try:
                    battery_str = drone_instance.send_command_with_return("battery?", timeout=5)
                    battery = int(battery_str)
                    if battery < 10:
                        logger.log_error('NavigationInterface', 
                            f'CRITICAL: Battery too low ({battery}%)')
                        return None
                except Exception as e:
                    logger.log_error('NavigationInterface', f'Battery check error: {e}')
                    return None
                
                prompt = f"\nSelect waypoint file (1-{len(files)}) or 'q' to quit: "
                print(prompt, end='', flush=True)

                # Linux: Use select for 5-second timeout
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
                        print("❌ Invalid input. Please enter a valid number.")
                else:
                    print("\r" + " " * 50 + "\r", end='')
                    continue
                    
            except Exception as e:
                logger.log_error('NavigationInterface', f'Error reading input: {e}')
                return None
    
    def _navigation_loop(self, drone_instance=None):
        """Interactive loop for waypoint selection and navigation."""
        loop_count = 0
        waypoints_history = []
        
        try:
            while self.is_running:
                # Display current position and available destinations
                print("\n" + "-" * 60)
                destinations = self.nav_manager.print_navigation_options()
                
                if not destinations:
                    logger.log_info('NavigationInterface', 
                        'No other waypoints to navigate to.')
                    break
                
                # Get user's navigation choice
                choice = self._get_navigation_choice(destinations, loop_count, drone_instance=drone_instance)
                
                if choice == 'quit':
                    logger.log_info('NavigationInterface', 'User chose to quit.')
                    break
                elif choice == 'reload':
                    logger.log_info('NavigationInterface', 'User chose to reload.')
                    if self._load_waypoint_file(drone_instance=drone_instance):
                        continue
                    else:
                        break
                elif isinstance(choice, str):
                    # Execute navigation to selected waypoint
                    logger.log_info('NavigationInterface', 
                        f'User selected waypoint: {choice}')
                    success = self.nav_manager.navigate_to_waypoint(choice, drone_instance=drone_instance)
                    if success:
                        logger.log_success('NavigationInterface', 
                            f'Navigation to {choice} completed!')
                        waypoints_history.append(choice)
                        loop_count += 1
                    else:
                        logger.log_error('NavigationInterface', 
                            f'Navigation to {choice} failed!')
                        break
                
        except Exception as e:
            logger.log_error('NavigationInterface', f'Error in navigation loop: {e}')
        finally:
            return waypoints_history
    
    def _get_navigation_choice(self, destinations: list, loopCount: int, drone_instance=None) -> str:
        """Display navigation menu and get user's waypoint choice."""
        print(f"\n🎮 NAVIGATION OPTIONS:")
        print("-" * 40)
        
        # Create numbered menu with type icons for super vs inner waypoints
        for i, (wp_id, wp_name, wp_type) in enumerate(destinations, 1):
            type_icon = "⭐" if wp_type == "super" else "📍"
            print(f"  {i}. {type_icon} Navigate to '{wp_name}' ({wp_id})")
        
        print(f"  r. Reload waypoint file")
        print(f"  q. Quit navigation")
        
        while True:
            try:
                # Battery monitoring
                try:
                    battery_str = drone_instance.send_command_with_return("battery?", timeout=5)
                    battery = int(battery_str)
                    if battery < 20:
                        logger.log_warning('NavigationInterface', 
                            f'Low battery: {battery}%')
                        if battery < 10:
                            logger.log_error('NavigationInterface', 
                                f'CRITICAL: Battery too low ({battery}%)')
                            return 'quit'
                except Exception as e:
                    logger.log_error('NavigationInterface', f'Battery check error: {e}')
                    return 'quit'
                
                if loopCount == 0:
                    prompt = f"\nEnter your choice (1-{len(destinations)}, r, q): "
                else:
                    prompt = f"\nEnter your choice (1-{len(destinations)}, q): "
                
                print(prompt, end='', flush=True)

                # Linux: Use select for 5-second timeout
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
                            print("❗ Can only reload at start of navigation.")
                            continue
                    else:
                        try:
                            waypoint_index = int(choice) - 1
                            if 0 <= waypoint_index < len(destinations):
                                return destinations[waypoint_index][0]  # Return waypoint ID
                            else:
                                print(f"❌ Invalid choice. Please enter 1-{len(destinations)}")
                        except ValueError:
                            print("❌ Invalid input. Please enter a valid number.")
                else:
                    print("\r" + " " * 50 + "\r", end='')
                    continue
                    
            except KeyboardInterrupt:
                return 'quit'
            except Exception as e:
                logger.log_error('NavigationInterface', f'Error reading input: {e}')
                return 'quit'
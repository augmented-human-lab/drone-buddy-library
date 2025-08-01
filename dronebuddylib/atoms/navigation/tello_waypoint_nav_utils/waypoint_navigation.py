#!/usr/bin/env python3
"""
Core waypoint navigation engine for autonomous drone movement.

This module provides the central navigation logic for executing waypoint-based drone navigation.
It loads waypoint maps created during manual flight sessions, calculates optimal paths between
waypoints, and executes precise movement sequences with yaw orientation and distance control.

Key Features:
- Bidirectional pathfinding (forward/reverse navigation)
- Movement reversal algorithms for return trips
- Yaw angle calculations for precise drone orientation
- Battery monitoring integration during navigation
- Emergency shutdown support for safety

Navigation Flow:
1. Load waypoint JSON file with movement sequences
2. Calculate path between current and target waypoint
3. Execute movement sequence with drone orientation control
4. Update current position after successful navigation
"""
import json
import time
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from dronebuddylib.utils.logger import Logger

logger = Logger()

class NavigationDirection(Enum):
    """Direction enum for waypoint navigation pathfinding."""
    FORWARD = "forward"    # Top-down navigation in waypoint sequence
    REVERSE = "reverse"    # Bottom-up navigation with reversed movements

@dataclass
class NavigationMovement:
    """
    Represents a single drone movement instruction with reversal capability.
    
    This class encapsulates individual movement commands (linear or vertical) with the ability
    to generate reversed movements for bidirectional navigation. Each movement includes distance,
    direction, and orientation data needed for precise drone control.
    """
    id: str                              # Unique identifier for the movement
    type: str                           # Movement type: "move" (horizontal) or "lift" (vertical)
    distance: float                     # Movement distance in centimeters
    direction: Optional[str] = None     # For "lift" type: "up" or "down"
    yaw: Optional[int] = None          # For "move" type: target yaw angle (-180 to 180)
    
    def reverse(self) -> 'NavigationMovement':
        """Create reversed movement for return navigation."""
        reversed_movement = NavigationMovement(
            id=str(uuid.uuid4()),        # New unique ID for reversed movement
            type=self.type,
            distance=self.distance,      # Same distance for reversed movement
            direction=self._reverse_direction(),
            yaw=self._reverse_yaw(),
        )
        return reversed_movement
    
    def _reverse_direction(self) -> Optional[str]:  
        """Reverse vertical movement direction (up <-> down)."""
        if self.type == "lift" and self.direction is not None:
            return "down" if self.direction == "up" else "up"
        else: 
            # For horizontal movements, direction is handled by yaw reversal
            return self.direction  # Keep same (None for move type)
    
    def _reverse_yaw(self) -> Optional[int]:
        """Calculate reverse yaw by adding 180 degrees."""
        if self.type == "move" and self.yaw is not None:
            # Reverse direction by adding 180 degrees
            reversed_raw = (self.yaw + 180)
            # Normalize to standard -180 to 180 degree range
            reversed_yaw = reversed_raw if reversed_raw <= 180 else reversed_raw - 360
            return reversed_yaw
        else:
            # For lift type, yaw is not applicable
            return self.yaw  # Keep same (None for lift type)

@dataclass
class Waypoint:
    """
    Represents a named waypoint with its associated movement sequence.
    
    Each waypoint contains the sequence of movements required to reach it from the previous
    waypoint, along with metadata for identification and sequencing.
    """
    id: str                                    # Unique waypoint identifier (e.g., "WP_001")
    name: str                                  # Human-readable waypoint name
    movements_to_here: List[NavigationMovement]  # Movement sequence to reach this waypoint
    index: int                                 # Position in the waypoint sequence

class WaypointNavigationManager:
    """
    Core navigation engine for autonomous waypoint-based drone navigation.
    
    This class manages the complete navigation workflow from waypoint file loading through
    path calculation and movement execution. It provides bidirectional navigation capabilities,
    allowing the drone to travel between any waypoints in either direction using optimized
    path planning algorithms.
    
    Features:
    - Waypoint file loading and parsing from JSON format
    - Forward and reverse pathfinding between waypoints
    - Movement sequence execution with precise yaw control
    - Battery monitoring integration during navigation
    - Emergency shutdown support for safety
    """
    
    def __init__(self, nav_speed: int, vertical_factor: float):
        """Initialize navigation manager with speed and vertical compensation settings."""
        logger.log_info('WaypointNavigationManager', 'Initializing waypoint navigation manager.')
        
        self.waypoints: Dict[str, Waypoint] = {}      # Waypoint storage by ID
        self.waypoint_order: List[str] = []           # Ordered sequence of waypoint IDs
        self.current_waypoint_id: str = "WP_001"     # Always start at first waypoint
        self.session_info: Dict = {}                  # Metadata from waypoint file
        self.json_file_path: str = ""                # Path to loaded waypoint file
        self.nav_speed = nav_speed                    # Movement speed for navigation
        self.vertical_factor = vertical_factor        # Compensation factor for vertical movements
        
        logger.log_debug('WaypointNavigationManager', f'Initialized with nav_speed={nav_speed}, vertical_factor={vertical_factor}')
    
    def load_waypoint_file(self, json_file_path: str) -> bool:
        """Load waypoints from JSON file and prepare navigation data."""
        try:
            logger.log_info('WaypointNavigationManager', f'Loading waypoint file: {json_file_path}')
            
            # Read and parse JSON waypoint data
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            
            self.json_file_path = json_file_path
            self.session_info = data.get('session_info', {})
            waypoints_data = data.get('waypoints', [])
            
            # Clear existing waypoint data
            self.waypoints.clear()
            self.waypoint_order.clear()
            
            # Process waypoints in sequential order
            for index, wp_data in enumerate(waypoints_data):
                movements = []
                
                # Convert movement data to NavigationMovement objects
                for mov_data in wp_data.get('movements_to_here', []):
                    movement = NavigationMovement(
                        id=mov_data['id'],
                        type=mov_data['type'],
                        direction=mov_data.get('direction', None),  # For lift movements
                        distance=mov_data['distance'],
                        yaw=mov_data.get('yaw', None)              # For horizontal movements
                    )
                    movements.append(movement)
                
                # Create waypoint object with movement sequence
                waypoint = Waypoint(
                    id=wp_data['id'],
                    name=wp_data['name'],
                    movements_to_here=movements,
                    index=index  # Sequence position for pathfinding
                )
                
                self.waypoints[waypoint.id] = waypoint
                self.waypoint_order.append(waypoint.id)
            
            # Reset navigation to starting position
            self.current_waypoint_id = "WP_001"
            
            logger.log_success('WaypointNavigationManager', f'Loaded {len(self.waypoints)} waypoints successfully.')
            self._print_waypoint_summary()
            
            return True
            
        except Exception as e:
            logger.log_error('WaypointNavigationManager', f'Error loading waypoint file: {e}')
            return False
    
    def _print_waypoint_summary(self):
        """Display formatted summary of all loaded waypoints."""
        print("\n📍 WAYPOINT SUMMARY")
        print("=" * 50)
        for wp_id in self.waypoint_order:
            waypoint = self.waypoints[wp_id]
            status = "🏠 CURRENT" if wp_id == self.current_waypoint_id else "  "
            print(f"{status} {waypoint.id}: '{waypoint.name}'")
        print("=" * 50)
    
    def get_available_destinations(self) -> List[Tuple[str, str]]:
        """Get list of waypoints available for navigation (excluding current position)."""
        destinations = []
        for wp_id in self.waypoint_order:
            if wp_id != self.current_waypoint_id:
                waypoint = self.waypoints[wp_id]
                destinations.append((wp_id, waypoint.name))
        return destinations
    
    def calculate_navigation_path(self, target_waypoint_id: str) -> Tuple[List[NavigationMovement], NavigationDirection]:
        """Calculate movement sequence and direction for navigation to target waypoint."""
        if target_waypoint_id not in self.waypoints:
            raise ValueError(f"Target waypoint {target_waypoint_id} not found")
        
        current_waypoint = self.waypoints[self.current_waypoint_id]
        target_waypoint = self.waypoints[target_waypoint_id]
        
        current_index = current_waypoint.index
        target_index = target_waypoint.index
        
        if target_index > current_index:
            # Forward navigation: follow waypoint sequence top-down
            return self._calculate_forward_path(current_index, target_index), NavigationDirection.FORWARD
        else:
            # Reverse navigation: use inverted movements bottom-up
            return self._calculate_reverse_path(current_index, target_index), NavigationDirection.REVERSE
    
    def _calculate_forward_path(self, current_waypoint_index: int, target_waypoint_index: int) -> List[NavigationMovement]:
        """Generate movement sequence for forward navigation."""
        movements = []
        
        # Collect movements from next waypoint through target waypoint
        for i in range(current_waypoint_index + 1, target_waypoint_index + 1):
            waypoint_id = self.waypoint_order[i]
            waypoint = self.waypoints[waypoint_id]
            movements.extend(waypoint.movements_to_here)
        
        return movements
    
    def _calculate_reverse_path(self, current_waypoint_index: int, target_waypoint_index: int) -> List[NavigationMovement]:
        """Generate movement sequence for reverse navigation with inverted movements."""
        movements = []
        
        # Process waypoints in reverse order from current back to target
        # Each movement is individually reversed (direction and yaw inverted)
        for i in range(current_waypoint_index, target_waypoint_index, -1):
            waypoint_id = self.waypoint_order[i]
            waypoint = self.waypoints[waypoint_id]
            
            # Reverse each movement and add in reverse order for proper sequencing
            reversed_movements = [mov.reverse() for mov in reversed(waypoint.movements_to_here)]
            movements.extend(reversed_movements)
        
        return movements
    
    def navigate_to_waypoint(self, target_waypoint_id: str, drone_instance=None) -> bool:
        """Execute complete navigation sequence to target waypoint with safety checks."""
        # Safety check for emergency shutdown (goto mode only)
        if hasattr(self, 'coordinator') and hasattr(self.coordinator, '_emergency_shutdown'):
            if self.coordinator._emergency_shutdown:
                logger.log_warning('WaypointNavigationManager', 'Emergency shutdown detected - aborting navigation')
                return False
                
        # Validate target waypoint exists
        if target_waypoint_id not in self.waypoints:
            logger.log_error('WaypointNavigationManager', f'Waypoint {target_waypoint_id} not found')
            return False
        
        # Check if already at target position
        if target_waypoint_id == self.current_waypoint_id:
            logger.log_info('WaypointNavigationManager', f'Already at waypoint {target_waypoint_id}')
            return True
        
        try:
            # Generate navigation plan with path calculation
            movements, direction = self.calculate_navigation_path(target_waypoint_id)
            target_name = self.waypoints[target_waypoint_id].name
            current_name = self.waypoints[self.current_waypoint_id].name
            
            logger.log_info('WaypointNavigationManager', f'Navigation plan: From {self.current_waypoint_id} ("{current_name}") to {target_waypoint_id} ("{target_name}") using {direction.value} direction with {len(movements)} movements')
            
            # Display navigation plan to user
            print(f"\n🧭 NAVIGATION PLAN")
            print(f"From: {self.current_waypoint_id} ('{current_name}')")
            print(f"To: {target_waypoint_id} ('{target_name}')")
            print(f"Direction: {direction.value}")
            print(f"Total movements: {len(movements)}")
            
            # Execute navigation movement sequence
            success = self._execute_navigation(movements, direction, drone_instance=drone_instance)
            
            if success:
                # Update current position after successful navigation
                self.current_waypoint_id = target_waypoint_id
                logger.log_success('WaypointNavigationManager', f'Successfully navigated to {target_waypoint_id} ("{target_name}")')
                return True
            else:
                logger.log_error('WaypointNavigationManager', f'Navigation to {target_waypoint_id} failed')
                return False
                
        except Exception as e:
            logger.log_error('WaypointNavigationManager', f'Navigation error: {e}')
            return False

    def _execute_navigation(self, movements: List[NavigationMovement], direction: NavigationDirection, drone_instance=None) -> bool:
        """Execute movement sequence with yaw control and safety monitoring."""
        
        logger.log_info('WaypointNavigationManager', f'Executing {len(movements)} movements ({direction.value})')
        
        # Pause battery monitoring to prevent command conflicts during navigation
        if hasattr(self, 'coordinator') and hasattr(self.coordinator, '_pause_battery_monitoring'):
            self.coordinator._pause_battery_monitoring()

        time.sleep(0.3)  # Allow battery monitoring to pause

        drone_instance.set_speed(self.nav_speed)  # Configure navigation speed
        
        try: 
            for i, movement in enumerate(movements, 1):
                # Emergency shutdown check before each movement (goto mode only)
                if hasattr(self, 'coordinator') and hasattr(self.coordinator, '_emergency_shutdown'):
                    if self.coordinator._emergency_shutdown:
                        logger.log_warning('WaypointNavigationManager', 'Emergency shutdown detected - stopping navigation execution')
                        return False

                # Battery safety check before each movement
                battery_str = drone_instance.send_command_with_return("battery?", timeout=3)
                logger.log_debug('WaypointNavigationManager', 'checking battery status')
                battery = int(battery_str)
                if battery < 20:
                    logger.log_warning('TelloWaypointNavCoordinator', f'Low battery detected: {battery}%')
                    if battery < 10:
                        logger.log_error('TelloWaypointNavCoordinator', f'CRITICAL: Battery too low ({battery}%), initiating emergency landing.')
                        return False
                
                logger.log_debug('WaypointNavigationManager', f'Step {i}/{len(movements)}: {movement.type} movement')
                
                # Ensure minimum movement distance for drone command validity
                distance = movement.distance if movement.distance is not None and movement.distance >= 20 else 20
                
                if movement.type == "move":
                    # Handle horizontal movement with yaw orientation
                    yaw = movement.yaw if movement.yaw is not None else 0
                    current_yaw = self.get_yaw(drone_instance=drone_instance)
                    
                    # Calculate required yaw adjustment
                    turn_degree = abs(yaw - current_yaw)
                    if current_yaw > yaw:
                        logger.log_debug('WaypointNavigationManager', f'Adjusting yaw from {current_yaw} to {yaw} degrees')
                        if turn_degree > 180 and turn_degree < 360: 
                            drone_instance.rotate_clockwise(360 - turn_degree)  # Shorter rotation path
                        elif turn_degree <= 180 and turn_degree > 0: 
                            drone_instance.rotate_counter_clockwise(turn_degree)
                        else: 
                            logger.log_debug('WaypointNavigationManager', 'No yaw adjustment needed')
                        drone_instance.send_rc_control(0, 0, 0, 0)  # Stop rotation
                    else: 
                        logger.log_debug('WaypointNavigationManager', f'Adjusting yaw from {current_yaw} to {yaw} degrees')
                        if turn_degree > 180 and turn_degree < 360: 
                            drone_instance.rotate_counter_clockwise(360 - turn_degree)  # Shorter rotation path
                        elif turn_degree <= 180 and turn_degree > 0: 
                            drone_instance.rotate_clockwise(turn_degree)
                        else: 
                            logger.log_debug('WaypointNavigationManager', 'No yaw adjustment needed')
                        drone_instance.send_rc_control(0, 0, 0, 0)  # Stop rotation

                    # Execute forward movement at target yaw
                    drone_instance.move_forward(int(distance))
                    drone_instance.send_rc_control(0, 0, 0, 0)  # Stop movement
                    logger.log_debug('WaypointNavigationManager', f'Moved forward {distance} cm at yaw {yaw} degrees')

                else:
                    # Handle vertical movements with direction-based compensation
                    match (movement.direction, direction): 
                        case ("up", NavigationDirection.FORWARD):
                            # Apply vertical compensation factor for upward movement
                            actual_distance = max((distance / self.vertical_factor), 20)
                            drone_instance.move_up(int(actual_distance))
                            logger.log_debug('WaypointNavigationManager', f'Lifted up {actual_distance} cm')
                        case ("down", NavigationDirection.FORWARD):
                            drone_instance.move_down(int(distance))
                            logger.log_debug('WaypointNavigationManager', f'Lowered down {distance} cm')
                        ############# REVERSE MOVEMENT HANDLING #############
                        case ("up", NavigationDirection.REVERSE):
                            drone_instance.move_up(int(distance))
                            logger.log_debug('WaypointNavigationManager', f'Lifted up {distance} cm')
                        case ("down", NavigationDirection.REVERSE):
                            # Apply compensation for reverse downward movement
                            actual_distance = max((distance / self.vertical_factor), 20)
                            drone_instance.move_down(int(actual_distance))
                            logger.log_debug('WaypointNavigationManager', f'Lowered down {actual_distance} cm')
                    
                    drone_instance.send_rc_control(0, 0, 0, 0)  # Stop vertical movement
            
            logger.log_success('WaypointNavigationManager', 'Navigation movements completed')
            return True
            
        except Exception as e:
            logger.log_error('WaypointNavigationManager', f'Error during navigation execution: {e}')
            drone_instance.send_rc_control(0, 0, 0, 0)  # Emergency stop
            return False
        finally:
            # Resume battery monitoring after navigation completion
            if hasattr(self, 'coordinator') and hasattr(self.coordinator, '_resume_battery_monitoring'):
                self.coordinator._resume_battery_monitoring()
    
    def get_yaw(self, drone_instance=None) -> int:
        """Get current drone yaw angle from attitude telemetry."""
        try:
            attitude_str = drone_instance.send_command_with_return("attitude?", timeout=3)
            logger.log_debug('WaypointNavigationManager', f'Raw attitude response: {attitude_str}')
            
            # Parse attitude string format: "pitch:0;roll:0;yaw:45;"
            yaw = 0  # Default fallback value
            if attitude_str and ':' in attitude_str:
                attitude_parts = attitude_str.split(';')
                for part in attitude_parts:
                    if part.strip() and 'yaw:' in part:
                        try:
                            yaw_value = part.split(':')[1].strip()
                            if yaw_value:
                                yaw = int(yaw_value)
                        except (ValueError, IndexError) as e:
                            logger.log_warning('WaypointNavigationManager', f'Failed to parse yaw from "{part}": {e}')
                            continue
            return yaw
        except Exception as e:
            logger.log_warning('WaypointNavigationManager', f'Attitude query failed: {e}')
            return 0  # Return default yaw on communication error
    
    def get_current_waypoint_info(self) -> Tuple[str, str]:
        """Get current waypoint ID and display name."""
        waypoint = self.waypoints[self.current_waypoint_id]
        return waypoint.id, waypoint.name
    
    def print_navigation_options(self):
        """Display formatted list of available navigation destinations."""
        destinations = self.get_available_destinations()
        current_id, current_name = self.get_current_waypoint_info()
        
        print(f"\n🏠 Current Position: {current_id} ('{current_name}')")
        print("\n📍 Available Destinations:")
        print("-" * 40)
        
        if not destinations:
            print("  No other waypoints available")
        else:
            for i, (wp_id, wp_name) in enumerate(destinations, 1):
                print(f"  {i}. {wp_id}: '{wp_name}'")
        
        return destinations
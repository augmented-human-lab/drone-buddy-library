#!/usr/bin/env python3
import json
import time
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class NavigationDirection(Enum):
    FORWARD = "forward"    # Top-down in waypoint file
    REVERSE = "reverse"    # Bottom-up in waypoint file

@dataclass
class NavigationMovement:
    """Represents a single movement instruction."""
    id: str
    type: str  # "move" or "lift"
    distance: float  
    direction: Optional[str] = None  # Only for lift type ("up" or "down")
    yaw: Optional[int] = None  # Only for move type
    
    def reverse(self) -> 'NavigationMovement':
        """Create a reversed version of this movement."""
        reversed_movement = NavigationMovement(
            id=str(uuid.uuid4()),
            type=self.type,
            distance=self.distance, 
            direction=self._reverse_direction(),
            yaw=self._reverse_yaw(),
        )
        return reversed_movement
    
    def _reverse_direction(self) -> Optional[str]:  
        """Reverse the lift direction."""
        if self.type == "lift" and self.direction is not None:
            return "down" if self.direction == "up" else "up"
        else: 
            # For move type, we'll reverse based on yaw angle
            return self.direction  # Keep same (None for move type), yaw handles the direction
    
    def _reverse_yaw(self) -> Optional[int]:
        """Reverse the yaw angle for move type."""
        if self.type == "move" and self.yaw is not None:
            # Reverse yaw by adding 180 degrees and keeping it within -180 to 180 range
            reversed_raw = (self.yaw + 180)
            reversed_yaw = reversed_raw if reversed_raw <= 180 else reversed_raw - 360
            return reversed_yaw
        else:
            # For lift type, yaw is not applicable
            return self.yaw  # Keep same (None for lift type)

@dataclass
class Waypoint:
    """Represents a waypoint with its movements."""
    id: str
    name: str
    movements_to_here: List[NavigationMovement]
    index: int  # Position in the waypoint sequence

class WaypointNavigationManager:
    """Manages waypoint navigation and pathfinding."""
    
    def __init__(self, nav_speed: int, vertical_factor: float):
        self.waypoints: Dict[str, Waypoint] = {}
        self.waypoint_order: List[str] = []  # Ordered list of waypoint IDs
        self.current_waypoint_id: str = "WP_001"  # Always start at START
        self.session_info: Dict = {}
        self.json_file_path: str = ""
        self.nav_speed = nav_speed  # Speed for navigation movements
        self.vertical_factor = vertical_factor  # Airflow factor for vertical movements
    
    def load_waypoint_file(self, json_file_path: str) -> bool:
        """Load waypoints from JSON file into memory."""
        try:
            print(f"📖 Loading waypoint file: {json_file_path}")
            
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            
            self.json_file_path = json_file_path
            self.session_info = data.get('session_info', {})
            waypoints_data = data.get('waypoints', [])
            
            # Clear existing data
            self.waypoints.clear()
            self.waypoint_order.clear()
            
            # Load waypoints in order
            for index, wp_data in enumerate(waypoints_data):
                movements = []
                for mov_data in wp_data.get('movements_to_here', []):
                    movement = NavigationMovement(
                        id=mov_data['id'],
                        type=mov_data['type'],
                        direction=mov_data.get('direction', None),
                        distance=mov_data['distance'],
                        yaw=mov_data.get('yaw', None)
                    )
                    movements.append(movement)
                
                waypoint = Waypoint(
                    id=wp_data['id'],
                    name=wp_data['name'],
                    movements_to_here=movements,
                    index=index
                )
                
                self.waypoints[waypoint.id] = waypoint
                self.waypoint_order.append(waypoint.id)
            
            # Reset to start position
            self.current_waypoint_id = "WP_001"
            
            print(f"✅ Loaded {len(self.waypoints)} waypoints successfully")
            self._print_waypoint_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading waypoint file: {e}")
            return False
    
    def _print_waypoint_summary(self):
        """Print a summary of loaded waypoints."""
        print("\n📍 WAYPOINT SUMMARY")
        print("=" * 50)
        for wp_id in self.waypoint_order:
            waypoint = self.waypoints[wp_id]
            status = "🏠 CURRENT" if wp_id == self.current_waypoint_id else "  "
            print(f"{status} {waypoint.id}: '{waypoint.name}'")
        print("=" * 50)
    
    def get_available_destinations(self) -> List[Tuple[str, str]]:
        """Get list of waypoints drone can navigate to (excluding current)."""
        destinations = []
        for wp_id in self.waypoint_order:
            if wp_id != self.current_waypoint_id:
                waypoint = self.waypoints[wp_id]
                destinations.append((wp_id, waypoint.name))
        return destinations
    
    def calculate_navigation_path(self, target_waypoint_id: str) -> Tuple[List[NavigationMovement], NavigationDirection]:
        """
        Calculate the movement sequence to navigate from current to target waypoint.
        
        Returns:
            Tuple of (movements_list, direction)
        """
        if target_waypoint_id not in self.waypoints:
            raise ValueError(f"Target waypoint {target_waypoint_id} not found")
        
        current_waypoint = self.waypoints[self.current_waypoint_id]
        target_waypoint = self.waypoints[target_waypoint_id]
        
        current_index = current_waypoint.index
        target_index = target_waypoint.index
        
        if target_index > current_index:
            # Forward navigation (top-down)
            return self._calculate_forward_path(current_index, target_index), NavigationDirection.FORWARD
        else:
            # Reverse navigation (bottom-up)
            return self._calculate_reverse_path(current_index, target_index), NavigationDirection.REVERSE
    
    def _calculate_forward_path(self, current_waypoint_index: int, target_waypoint_index: int) -> List[NavigationMovement]:
        """Calculate forward navigation path (normal order)."""
        movements = []
        
        # Collect all movements from next waypoint to target waypoint
        for i in range(current_waypoint_index + 1, target_waypoint_index + 1):
            waypoint_id = self.waypoint_order[i]
            waypoint = self.waypoints[waypoint_id]
            movements.extend(waypoint.movements_to_here)
        
        return movements
    
    def _calculate_reverse_path(self, current_waypoint_index: int, target_waypoint_index: int) -> List[NavigationMovement]:
        """Calculate reverse navigation path (reversed movements)."""
        movements = []
        
        # Collect all movements from current waypoint back to target waypoint
        # Reverse the order AND reverse each individual movement
        for i in range(current_waypoint_index, target_waypoint_index, -1):
            waypoint_id = self.waypoint_order[i]
            waypoint = self.waypoints[waypoint_id]
            
            # Reverse each movement and add to list in reverse order
            reversed_movements = [mov.reverse() for mov in reversed(waypoint.movements_to_here)]
            movements.extend(reversed_movements)
        
        return movements
    
    def navigate_to_waypoint(self, target_waypoint_id: str, drone_instance=None) -> bool:
        """
        Navigate to target waypoint and update current position.
        
        Returns:
            True if navigation successful, False otherwise
        """
        if target_waypoint_id not in self.waypoints:
            print(f"❌ Waypoint {target_waypoint_id} not found")
            return False
        
        if target_waypoint_id == self.current_waypoint_id:
            print(f"ℹ️  Already at waypoint {target_waypoint_id}")
            return True
        
        try:
            # Calculate navigation path
            movements, direction = self.calculate_navigation_path(target_waypoint_id)
            target_name = self.waypoints[target_waypoint_id].name
            current_name = self.waypoints[self.current_waypoint_id].name
            
            print(f"\n🧭 NAVIGATION PLAN")
            print(f"From: {self.current_waypoint_id} ('{current_name}')")
            print(f"To: {target_waypoint_id} ('{target_name}')")
            print(f"Direction: {direction.value}")
            print(f"Total movements: {len(movements)}")
            
            # Execute navigation
            success = self._execute_navigation(movements, direction, drone_instance=drone_instance)
            
            if success:
                # Update current position
                self.current_waypoint_id = target_waypoint_id
                print(f"✅ Successfully navigated to {target_waypoint_id} ('{target_name}')")
                return True
            else:
                print(f"❌ Navigation to {target_waypoint_id} failed")
                return False
                
        except Exception as e:
            print(f"❌ Navigation error: {e}")
            return False

    def _execute_navigation(self, movements: List[NavigationMovement], direction: NavigationDirection, drone_instance=None) -> bool:
        """Execute the navigation movements."""
        
        print(f"\n🚁 Executing {len(movements)} movements ({direction.value})...")
        drone_instance.set_speed(self.nav_speed)  # Set navigation speed
        try: 
            for i, movement in enumerate(movements, 1):
                print(f"  Step {i}/{len(movements)}: {movement.type} ")
                distance = movement.distance if movement.distance is not None and movement.distance >= 20 else 20  # Ensure minimum valid distance for movement
                if movement.type == "move":
                    yaw = movement.yaw if movement.yaw is not None else 0
                    current_yaw = self.get_yaw(drone_instance=drone_instance)
                    
                    turn_degree = abs(yaw - current_yaw)
                    if current_yaw > yaw:
                        print(f"  Adjusting yaw from {current_yaw} to {yaw} degrees")
                        if turn_degree > 180 and turn_degree < 360: 
                            drone_instance.rotate_clockwise(360 - turn_degree)
                        elif turn_degree <= 180 and turn_degree > 0: 
                            drone_instance.rotate_counter_clockwise(turn_degree)
                        else: 
                            print("  No yaw adjustment needed")
                        drone_instance.send_rc_control(0, 0, 0, 0)  # Stop any ongoing movement
                    else: 
                        print(f"  Adjusting yaw from {current_yaw} to {yaw} degrees")
                        if turn_degree > 180 and turn_degree < 360: 
                            drone_instance.rotate_counter_clockwise(360 - turn_degree)
                        elif turn_degree <= 180 and turn_degree > 0: 
                            drone_instance.rotate_clockwise(turn_degree)
                        else: 
                            print("  No yaw adjustment needed")
                        drone_instance.send_rc_control(0, 0, 0, 0)  # Stop any ongoing movement

                    drone_instance.move_forward(int(distance))
                    time.sleep(0.5)  # Allow some time for the drone to stabilize
                    drone_instance.send_rc_control(0, 0, 0, 0)  # Stop any ongoing movement
                    print(f"  Moved forward {distance} cm at yaw {yaw} degrees")

                else:
                    match (movement.direction, direction): 
                        case ("up", NavigationDirection.FORWARD):
                            actual_distance = max((distance / self.vertical_factor), 20)
                            drone_instance.move_up(int(actual_distance))
                            print(f"  Lifted up {actual_distance} cm")
                        case ("down", NavigationDirection.FORWARD):
                            drone_instance.move_down(int(distance))
                            print(f"  Lowered down {distance} cm")
                        ############# REVERSE MOVEMENT HANDLING #############
                        case ("up", NavigationDirection.REVERSE):
                            drone_instance.move_up(int(distance))
                            print(f"  Lifted up {distance} cm")
                        case ("down", NavigationDirection.REVERSE):
                            actual_distance = max((distance / self.vertical_factor), 20)
                            drone_instance.move_down(int(actual_distance))
                            print(f"  Lowered down {actual_distance} cm")
                    
                    drone_instance.send_rc_control(0, 0, 0, 0)  # Stop any ongoing movement
            
            print("✅ Navigation movements completed")
            return True
        except Exception as e:
            print(f"❌ Error during navigation execution: {e}")
            drone_instance.send_rc_control(0, 0, 0, 0)  # Stop any ongoing movement
            return False
    
    def get_yaw(self, drone_instance=None) -> int:
        try:
            attitude_str = drone_instance.send_command_with_return("attitude?", timeout=3)
            print(f"Raw attitude response: '{attitude_str}'")  # Debug line
            
            # Parse attitude string like "pitch:0;roll:0;yaw:45;"
            yaw = 0  # Default value
            if attitude_str and ':' in attitude_str:
                attitude_parts = attitude_str.split(';')
                for part in attitude_parts:
                    if part.strip() and 'yaw:' in part:
                        try:
                            yaw_value = part.split(':')[1].strip()
                            if yaw_value:
                                yaw = int(yaw_value)
                        except (ValueError, IndexError) as e:
                            print(f"⚠️  Failed to parse yaw from '{part}': {e}")
                            continue
            return yaw
        except Exception as e:
            print(f"⚠️  Attitude query failed: {e}")
            return 0
    
    def get_current_waypoint_info(self) -> Tuple[str, str]:
        """Get current waypoint ID and name."""
        waypoint = self.waypoints[self.current_waypoint_id]
        return waypoint.id, waypoint.name
    
    def print_navigation_options(self):
        """Print available navigation destinations."""
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
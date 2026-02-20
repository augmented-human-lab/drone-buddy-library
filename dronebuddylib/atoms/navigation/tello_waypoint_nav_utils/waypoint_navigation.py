#!/usr/bin/env python3
"""
Core 2D waypoint navigation engine for smart autonomous drone movement.

This module provides hierarchical navigation logic for executing waypoint-based drone navigation
using a 2D Super Waypoint / Inner Waypoint architecture. It enables smart routing between
any two waypoints by navigating through Super Waypoint hubs.

Key Features:
- 2D hierarchical waypoint structure (Super Waypoints + Inner Waypoints)
- Smart pathfinding through Super Waypoint hubs
- Mixed forward/reverse movement sequences in single navigation
- Inner waypoint to Super Waypoint reverse navigation
- Super Waypoint to Super Waypoint bidirectional navigation
- MiDaS depth-based obstacle detection before forward movements

Navigation Flow:
1. Load 2D waypoint JSON file with Super Waypoints and Inner Waypoints
2. Determine if current/target are Super or Inner waypoints
3. Calculate smart path: Inner -> SuperWP -> ... -> SuperWP -> Inner (if needed)
4. Execute mixed movement sequences with proper forward/reverse handling
5. Check for obstacles using MiDaS depth estimation before each forward movement
6. Update current position after successful navigation
"""
import json
import time
import uuid
import threading
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from dronebuddylib.utils.logger import Logger

logger = Logger()


class NavigationDirection(Enum):
    """Direction enum for waypoint navigation pathfinding."""
    FORWARD = "forward"    # Normal order navigation
    REVERSE = "reverse"    # Reversed movements navigation


@dataclass
class NavigationMovement:
    """
    Represents a single drone movement instruction with reversal capability.
    """
    id: str
    type: str                           # "move" (horizontal) or "lift" (vertical)
    distance: float
    direction: Optional[str] = None     # For "lift" type: "up" or "down"
    yaw: Optional[int] = None           # For "move" type: target yaw angle
    
    def reverse(self) -> 'NavigationMovement':
        """Create reversed movement for return navigation."""
        return NavigationMovement(
            id=str(uuid.uuid4()),
            type=self.type,
            distance=self.distance,
            direction=self._reverse_direction(),
            yaw=self._reverse_yaw(),
        )
    
    def _reverse_direction(self) -> Optional[str]:
        """Reverse vertical movement direction (up <-> down)."""
        if self.type == "lift" and self.direction is not None:
            return "down" if self.direction == "up" else "up"
        return self.direction
    
    def _reverse_yaw(self) -> Optional[int]:
        """Calculate reverse yaw by adding 180 degrees."""
        if self.type == "move" and self.yaw is not None:
            reversed_raw = (self.yaw + 180)
            return reversed_raw if reversed_raw <= 180 else reversed_raw - 360
        return self.yaw


@dataclass
class InnerWaypoint:
    """Represents an inner waypoint within a Super Waypoint's local network."""
    id: str
    name: str
    parent_super_waypoint_id: str       # ID of the associated Super Waypoint
    movements_to_here: List[NavigationMovement] = field(default_factory=list)


@dataclass
class SuperWaypoint:
    """Represents a Super Waypoint hub."""
    id: str
    name: str
    index: int                          # Position in Super Waypoint sequence
    movements_to_here: List[NavigationMovement] = field(default_factory=list)
    inner_waypoints: Dict[str, InnerWaypoint] = field(default_factory=dict)


@dataclass
class NavigationSegment:
    """Represents a segment of navigation with its direction."""
    movements: List[NavigationMovement]
    direction: NavigationDirection
    description: str


class WaypointNavigationManager:
    """
    Core 2D navigation engine for smart waypoint-based drone navigation.
    
    This class manages navigation through a 2D hierarchical waypoint structure,
    enabling smart routing between any two waypoints by navigating through
    Super Waypoint hubs.
    
    Navigation Logic:
    - To go from any waypoint A to any waypoint B:
      1. If at Inner Waypoint: Reverse to associated Super Waypoint
      2. Navigate between Super Waypoints (forward or reverse direction)
      3. If target is Inner Waypoint: Forward from Super Waypoint to target
    """
    
    # Rotation compensation factor (degrees) - adjust if drone under/over-rotates
    # Positive value = rotate more, Negative value = rotate less
    ROTATION_COMPENSATION = 2  # Default: no compensation. Try 1-3 if drone under-rotates
    
    # Obstacle check retry settings
    OBSTACLE_CHECK_INTERVAL = 0.5  # Seconds between obstacle checks when blocked
    OBSTACLE_CHECK_MAX_WAIT = 30.0  # Maximum seconds to wait for obstacle to clear
    
    def __init__(self, nav_speed: int, vertical_factor: float, 
                 obstacle_detector=None, frame_read=None):
        """
        Initialize navigation manager with speed settings and optional obstacle detection.
        
        Args:
            nav_speed: Navigation speed in cm/s
            vertical_factor: Vertical movement compensation factor
            obstacle_detector: MiDaSObstacleDetector instance for depth-based obstacle detection
            frame_read: Tello frame reader for video feed access
        """
        logger.log_info('WaypointNavigationManager', 'Initializing 2D waypoint navigation manager.')
        
        self.super_waypoints: Dict[str, SuperWaypoint] = {}
        self.super_waypoint_order: List[str] = []  # Ordered sequence of Super Waypoint IDs
        self.all_waypoints: Dict[str, Union[SuperWaypoint, InnerWaypoint]] = {}  # All waypoints by ID
        
        self.current_waypoint_id: str = ""  # Current position (can be Super or Inner)
        self.current_super_waypoint_id: str = ""  # Current associated Super Waypoint
        
        self.session_info: Dict = {}
        self.json_file_path: str = ""
        self.nav_speed = nav_speed
        self.vertical_factor = vertical_factor
        
        # Obstacle detection components
        self.obstacle_detector = obstacle_detector
        self.frame_read = frame_read
        
        # Video display components
        self.video_display_thread = None
        self.video_display_running = False
        self.latest_depth_frame = None
        self._depth_frame_lock = threading.Lock()
        
        # External frame callback for GUI integration
        # When set, video frames are sent to this callback instead of cv2.imshow
        # Callback signature: (camera_frame_bgr, depth_frame_bgr) -> None
        self._external_frame_callback = None
        self._external_frame_callback_lock = threading.Lock()
        
        # Pause flag for obstacle detection
        # Paused by default - only runs during forward movement obstacle checks
        # This saves CPU/GPU resources when drone is hovering or waiting for commands
        self._obstacle_detection_paused = True
        
        logger.log_debug('WaypointNavigationManager', 
            f'Initialized with nav_speed={nav_speed}, vertical_factor={vertical_factor}, '
            f'obstacle_detection={"enabled" if obstacle_detector else "disabled"}')
    
    def load_waypoint_file(self, json_file_path: str) -> bool:
        """Load 2D waypoints from JSON file."""
        try:
            logger.log_info('WaypointNavigationManager', f'Loading waypoint file: {json_file_path}')
            
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            
            self.json_file_path = json_file_path
            self.session_info = data.get('session_info', {})
            
            # Check for 2D format
            format_version = self.session_info.get('format_version', '1.0')
            if format_version != '2.0':
                logger.log_warning('WaypointNavigationManager', 
                    f'Waypoint file format is {format_version}, expected 2.0. Attempting to load anyway.')
            
            super_waypoints_data = data.get('super_waypoints', [])
            
            # Clear existing data
            self.super_waypoints.clear()
            self.super_waypoint_order.clear()
            self.all_waypoints.clear()
            
            # Process Super Waypoints
            for index, swp_data in enumerate(super_waypoints_data):
                # Parse movements to Super Waypoint
                movements = self._parse_movements(swp_data.get('movements_to_here', []))
                
                # Create Super Waypoint
                super_wp = SuperWaypoint(
                    id=swp_data['id'],
                    name=swp_data['name'],
                    index=index,
                    movements_to_here=movements,
                    inner_waypoints={}
                )
                
                # Process Inner Waypoints
                for iwp_data in swp_data.get('inner_waypoints', []):
                    inner_movements = self._parse_movements(iwp_data.get('movements_to_here', []))
                    
                    inner_wp = InnerWaypoint(
                        id=iwp_data['id'],
                        name=iwp_data['name'],
                        parent_super_waypoint_id=super_wp.id,
                        movements_to_here=inner_movements
                    )
                    
                    super_wp.inner_waypoints[inner_wp.id] = inner_wp
                    self.all_waypoints[inner_wp.id] = inner_wp
                
                self.super_waypoints[super_wp.id] = super_wp
                self.super_waypoint_order.append(super_wp.id)
                self.all_waypoints[super_wp.id] = super_wp
            
            # Set current position to first Super Waypoint
            if self.super_waypoint_order:
                self.current_waypoint_id = self.super_waypoint_order[0]
                self.current_super_waypoint_id = self.super_waypoint_order[0]
            
            logger.log_success('WaypointNavigationManager', 
                f'Loaded {len(self.super_waypoints)} Super Waypoints successfully.')
            self._print_waypoint_summary()
            
            return True
            
        except Exception as e:
            logger.log_error('WaypointNavigationManager', f'Error loading waypoint file: {e}')
            import traceback
            traceback.print_exc()
            return False
    
    def _parse_movements(self, movements_data: list) -> List[NavigationMovement]:
        """Parse movement data from JSON into NavigationMovement objects."""
        movements = []
        for mov_data in movements_data:
            movement = NavigationMovement(
                id=mov_data['id'],
                type=mov_data['type'],
                direction=mov_data.get('direction', None),
                distance=mov_data['distance'],
                yaw=mov_data.get('yaw', None)
            )
            movements.append(movement)
        return movements
    
    def _print_waypoint_summary(self):
        """Display formatted summary of all loaded waypoints."""
        print("\n" + "=" * 60)
        print("📍 2D WAYPOINT STRUCTURE")
        print("=" * 60)
        
        for swp_id in self.super_waypoint_order:
            swp = self.super_waypoints[swp_id]
            current_marker = "🏠 CURRENT" if swp_id == self.current_waypoint_id else "  "
            print(f"\n{current_marker} ⭐ {swp.id}: '{swp.name}' (Super Waypoint)")
            
            for iwp_id, iwp in swp.inner_waypoints.items():
                current_marker = "🏠 CURRENT" if iwp_id == self.current_waypoint_id else "  "
                print(f"   {current_marker} └── 📍 {iwp.id}: '{iwp.name}'")
        
        print("=" * 60)
    
    def get_available_destinations(self) -> List[Tuple[str, str, str]]:
        """
        Get list of all waypoints available for navigation.
        
        Returns:
            List of tuples: (waypoint_id, waypoint_name, waypoint_type)
        """
        destinations = []
        
        for swp_id in self.super_waypoint_order:
            swp = self.super_waypoints[swp_id]
            
            if swp_id != self.current_waypoint_id:
                destinations.append((swp_id, swp.name, "super"))
            
            for iwp_id, iwp in swp.inner_waypoints.items():
                if iwp_id != self.current_waypoint_id:
                    destinations.append((iwp_id, iwp.name, "inner"))
        
        return destinations
    
    def _is_super_waypoint(self, waypoint_id: str) -> bool:
        """Check if waypoint ID belongs to a Super Waypoint."""
        return waypoint_id in self.super_waypoints
    
    def _get_parent_super_waypoint(self, waypoint_id: str) -> Optional[str]:
        """Get the parent Super Waypoint ID for any waypoint."""
        if self._is_super_waypoint(waypoint_id):
            return waypoint_id
        
        waypoint = self.all_waypoints.get(waypoint_id)
        if isinstance(waypoint, InnerWaypoint):
            return waypoint.parent_super_waypoint_id
        
        return None
    
    def calculate_navigation_path(self, target_waypoint_id: str) -> List[NavigationSegment]:
        """
        Calculate the complete navigation path from current position to target.
        
        Returns a list of NavigationSegments, each containing movements and direction.
        The path may involve:
        1. Reverse from current Inner Waypoint to Super Waypoint (if at Inner)
        2. Navigate between Super Waypoints (forward or reverse)
        3. Forward from Super Waypoint to target Inner Waypoint (if target is Inner)
        """
        if target_waypoint_id not in self.all_waypoints:
            raise ValueError(f"Target waypoint {target_waypoint_id} not found")
        
        if target_waypoint_id == self.current_waypoint_id:
            return []  # Already at target
        
        segments = []
        
        # Get Super Waypoint associations
        current_swp_id = self._get_parent_super_waypoint(self.current_waypoint_id)
        target_swp_id = self._get_parent_super_waypoint(target_waypoint_id)
        
        current_is_super = self._is_super_waypoint(self.current_waypoint_id)
        target_is_super = self._is_super_waypoint(target_waypoint_id)
        
        logger.log_debug('WaypointNavigationManager', 
            f'Path calculation: {self.current_waypoint_id} -> {target_waypoint_id}')
        logger.log_debug('WaypointNavigationManager', 
            f'Current SWP: {current_swp_id}, Target SWP: {target_swp_id}')
        logger.log_debug('WaypointNavigationManager', 
            f'Current is Super: {current_is_super}, Target is Super: {target_is_super}')
        
        # STEP 1: If at Inner Waypoint, reverse to Super Waypoint
        if not current_is_super:
            inner_wp = self.all_waypoints[self.current_waypoint_id]
            if isinstance(inner_wp, InnerWaypoint) and inner_wp.movements_to_here:
                # Reverse movements to go back to Super Waypoint
                reversed_movements = [mov.reverse() for mov in reversed(inner_wp.movements_to_here)]
                segments.append(NavigationSegment(
                    movements=reversed_movements,
                    direction=NavigationDirection.REVERSE,
                    description=f"Return from '{inner_wp.name}' to Super Waypoint '{self.super_waypoints[current_swp_id].name}'"
                ))
        
        # STEP 2: Navigate between Super Waypoints (if needed)
        if current_swp_id != target_swp_id:
            current_swp_index = self.super_waypoints[current_swp_id].index
            target_swp_index = self.super_waypoints[target_swp_id].index
            
            if target_swp_index > current_swp_index:
                # Forward navigation through Super Waypoints
                swp_segments = self._calculate_forward_super_path(current_swp_index, target_swp_index)
                segments.extend(swp_segments)
            else:
                # Reverse navigation through Super Waypoints
                swp_segments = self._calculate_reverse_super_path(current_swp_index, target_swp_index)
                segments.extend(swp_segments)
        
        # STEP 3: If target is Inner Waypoint, forward from Super Waypoint
        if not target_is_super:
            inner_wp = self.all_waypoints[target_waypoint_id]
            if isinstance(inner_wp, InnerWaypoint) and inner_wp.movements_to_here:
                segments.append(NavigationSegment(
                    movements=inner_wp.movements_to_here.copy(),
                    direction=NavigationDirection.FORWARD,
                    description=f"Navigate to Inner Waypoint '{inner_wp.name}'"
                ))
        
        return segments
    
    def _calculate_forward_super_path(self, current_index: int, target_index: int) -> List[NavigationSegment]:
        """Generate navigation segments for forward Super Waypoint traversal."""
        segments = []
        
        for i in range(current_index + 1, target_index + 1):
            swp_id = self.super_waypoint_order[i]
            swp = self.super_waypoints[swp_id]
            
            if swp.movements_to_here:
                segments.append(NavigationSegment(
                    movements=swp.movements_to_here.copy(),
                    direction=NavigationDirection.FORWARD,
                    description=f"Navigate to Super Waypoint '{swp.name}'"
                ))
        
        return segments
    
    def _calculate_reverse_super_path(self, current_index: int, target_index: int) -> List[NavigationSegment]:
        """Generate navigation segments for reverse Super Waypoint traversal."""
        segments = []
        
        # Process Super Waypoints in reverse order
        for i in range(current_index, target_index, -1):
            swp_id = self.super_waypoint_order[i]
            swp = self.super_waypoints[swp_id]
            
            if swp.movements_to_here:
                # Reverse each movement and reverse the order
                reversed_movements = [mov.reverse() for mov in reversed(swp.movements_to_here)]
                
                # Description references the previous Super Waypoint (destination)
                prev_swp_id = self.super_waypoint_order[i - 1]
                prev_swp = self.super_waypoints[prev_swp_id]
                
                segments.append(NavigationSegment(
                    movements=reversed_movements,
                    direction=NavigationDirection.REVERSE,
                    description=f"Navigate back to Super Waypoint '{prev_swp.name}'"
                ))
        
        return segments
    
    def navigate_to_waypoint(self, target_waypoint_id: str, drone_instance=None) -> bool:
        """Execute complete navigation sequence to target waypoint."""
        # Safety check for emergency shutdown
        if hasattr(self, 'coordinator') and hasattr(self.coordinator, '_emergency_shutdown'):
            if self.coordinator._emergency_shutdown:
                logger.log_warning('WaypointNavigationManager', 
                    'Emergency shutdown detected - aborting navigation')
                return False
        
        # Validate target
        if target_waypoint_id not in self.all_waypoints:
            logger.log_error('WaypointNavigationManager', 
                f'Waypoint {target_waypoint_id} not found')
            return False
        
        # Check if already at target
        if target_waypoint_id == self.current_waypoint_id:
            logger.log_info('WaypointNavigationManager', 
                f'Already at waypoint {target_waypoint_id}')
            return True
        
        try:
            # Calculate navigation path
            segments = self.calculate_navigation_path(target_waypoint_id)
            
            target_wp = self.all_waypoints[target_waypoint_id]
            target_name = target_wp.name
            current_wp = self.all_waypoints[self.current_waypoint_id]
            current_name = current_wp.name
            
            # Count total movements
            total_movements = sum(len(seg.movements) for seg in segments)
            
            logger.log_info('WaypointNavigationManager', 
                f'Navigation plan: {self.current_waypoint_id} -> {target_waypoint_id} '
                f'with {len(segments)} segments, {total_movements} total movements')
            
            # Display navigation plan
            print(f"\n" + "=" * 60)
            print("🧭 2D NAVIGATION PLAN")
            print("=" * 60)
            print(f"From: {self.current_waypoint_id} ('{current_name}')")
            print(f"To:   {target_waypoint_id} ('{target_name}')")
            print(f"Total segments: {len(segments)}")
            print(f"Total movements: {total_movements}")
            print("-" * 60)
            
            for i, seg in enumerate(segments, 1):
                direction_icon = "⬆️" if seg.direction == NavigationDirection.FORWARD else "⬇️"
                print(f"  {i}. {direction_icon} {seg.description} ({len(seg.movements)} moves)")
            
            print("=" * 60)
            
            # Execute navigation segments
            success = self._execute_navigation_segments(segments, drone_instance)
            
            if success:
                # Update current position
                self.current_waypoint_id = target_waypoint_id
                self.current_super_waypoint_id = self._get_parent_super_waypoint(target_waypoint_id)
                
                logger.log_success('WaypointNavigationManager', 
                    f'Successfully navigated to {target_waypoint_id} ("{target_name}")')
                return True
            else:
                logger.log_error('WaypointNavigationManager', 
                    f'Navigation to {target_waypoint_id} failed')
                return False
                
        except Exception as e:
            logger.log_error('WaypointNavigationManager', f'Navigation error: {e}')
            import traceback
            traceback.print_exc()
            return False
    
    def _execute_navigation_segments(self, segments: List[NavigationSegment], drone_instance=None) -> bool:
        """Execute all navigation segments in sequence."""
        logger.log_info('WaypointNavigationManager', 
            f'Executing {len(segments)} navigation segments')
        
        # Pause battery monitoring
        if hasattr(self, 'coordinator') and hasattr(self.coordinator, '_pause_battery_monitoring'):
            self.coordinator._pause_battery_monitoring()
        
        time.sleep(0.3)
        drone_instance.set_speed(self.nav_speed)
        
        try:
            for seg_idx, segment in enumerate(segments, 1):
                print(f"\n▶️ Segment {seg_idx}/{len(segments)}: {segment.description}")
                
                success = self._execute_single_segment(segment, drone_instance)
                
                if not success:
                    logger.log_error('WaypointNavigationManager', 
                        f'Segment {seg_idx} failed: {segment.description}')
                    return False
                
                print(f"  ✅ Segment {seg_idx} complete")
            
            logger.log_success('WaypointNavigationManager', 'All navigation segments completed')
            return True
            
        except Exception as e:
            logger.log_error('WaypointNavigationManager', f'Error during navigation: {e}')
            drone_instance.send_rc_control(0, 0, 0, 0)
            return False
        finally:
            if hasattr(self, 'coordinator') and hasattr(self.coordinator, '_resume_battery_monitoring'):
                self.coordinator._resume_battery_monitoring()
    
    def _execute_single_segment(self, segment: NavigationSegment, drone_instance=None) -> bool:
        """Execute a single navigation segment (list of movements) with obstacle detection."""
        try:
            for i, movement in enumerate(segment.movements, 1):
                # Emergency shutdown check
                if hasattr(self, 'coordinator') and hasattr(self.coordinator, '_emergency_shutdown'):
                    if self.coordinator._emergency_shutdown:
                        logger.log_warning('WaypointNavigationManager', 
                            'Emergency shutdown detected')
                        return False
                
                # Battery check
                try:
                    battery_str = drone_instance.send_command_with_return("battery?", timeout=3)
                    logger.log_debug('WaypointNavigationManager', 'Checking battery status')
                    battery = int(battery_str)
                    if battery < 20:
                        logger.log_warning('WaypointNavigationManager', f'Low battery detected: {battery}%')
                        if battery < 10:
                            logger.log_error('WaypointNavigationManager', 
                                f'CRITICAL: Battery too low ({battery}%), initiating emergency landing.')
                            return False
                except:
                    pass
                
                logger.log_debug('WaypointNavigationManager', 
                    f'Movement {i}/{len(segment.movements)}: {movement.type}')
                
                # Ensure minimum distance
                distance = max(movement.distance, 20) if movement.distance is not None else 20
                
                if movement.type == "move":
                    # Handle horizontal movement with yaw orientation
                    yaw = movement.yaw if movement.yaw is not None else 0
                    current_yaw = self.get_yaw(drone_instance)
                    
                    # Calculate yaw adjustment with compensation
                    turn_degree = abs(yaw - current_yaw)
                    compensation = self.ROTATION_COMPENSATION
                    
                    if current_yaw > yaw:
                        if turn_degree > 180 and turn_degree < 360:
                            drone_instance.rotate_clockwise(360 - turn_degree + compensation)
                        elif turn_degree <= 180 and turn_degree > 0:
                            drone_instance.rotate_counter_clockwise(turn_degree + compensation)
                    else:
                        if turn_degree > 180 and turn_degree < 360:
                            drone_instance.rotate_counter_clockwise(360 - turn_degree + compensation)
                        elif turn_degree <= 180 and turn_degree > 0:
                            drone_instance.rotate_clockwise(turn_degree + compensation)
                    
                    drone_instance.send_rc_control(0, 0, 0, 0)
                    
                    # OBSTACLE DETECTION: Temporarily enable MiDaS, check for obstacles, then disable
                    # This ensures MiDaS only runs when needed (before forward movements)
                    self._obstacle_detection_paused = False  # Enable MiDaS inference
                    path_clear = self._wait_for_clear_path(drone_instance)
                    self._obstacle_detection_paused = True   # Disable MiDaS inference after check
                    
                    if not path_clear:
                        logger.log_error('WaypointNavigationManager', 
                            'Obstacle detection timeout - path not clear')
                        # Set flag to indicate obstacle timeout specifically
                        from dronebuddylib.atoms.navigation.tello_waypoint_nav_utils.tello_waypoint_nav_coordinator import TelloWaypointNavCoordinator
                        TelloWaypointNavCoordinator._obstacle_timeout_occurred = True
                        return False
                    
                    # Execute forward movement
                    drone_instance.move_forward(int(distance))
                    drone_instance.send_rc_control(0, 0, 0, 0)
                    
                else:  # lift movement
                    # Handle vertical with direction-based compensation
                    if segment.direction == NavigationDirection.FORWARD:
                        if movement.direction == "up":
                            actual_distance = max(distance / self.vertical_factor, 20)
                            drone_instance.move_up(int(actual_distance))
                        elif movement.direction == "down":
                            drone_instance.move_down(int(distance))
                    else:  # REVERSE
                        if movement.direction == "up":
                            drone_instance.move_up(int(distance))
                        elif movement.direction == "down":
                            actual_distance = max(distance / self.vertical_factor, 20)
                            drone_instance.move_down(int(actual_distance))
                    
                    drone_instance.send_rc_control(0, 0, 0, 0)
            
            return True
            
        except Exception as e:
            logger.log_error('WaypointNavigationManager', f'Error executing segment: {e}')
            drone_instance.send_rc_control(0, 0, 0, 0)
            return False
    
    def _wait_for_clear_path(self, drone_instance=None) -> bool:
        """
        Check for obstacles using MiDaS depth estimation and wait until path is clear.
        
        This method is called before each forward movement. If an obstacle is detected,
        it will keep checking at regular intervals until the path is clear or timeout.
        Sends keep-alive commands to prevent drone from landing during extended waits.
        
        Returns:
            bool: True if path is clear (or obstacle detection disabled), False if timeout
        """
        # Skip if obstacle detection is not configured
        if self.obstacle_detector is None or self.frame_read is None:
            return True
        
        # Import here to avoid circular imports
        from dronebuddylib.models.enums import ObstacleDetectionMode
        
        # Skip if detection mode is OFF
        if self.obstacle_detector.detection_mode == ObstacleDetectionMode.OFF:
            return True
        
        start_time = time.time()
        last_keepalive_time = time.time()
        check_count = 0
        KEEPALIVE_INTERVAL = 5.0  # Send keep-alive every 5 seconds to prevent 15s timeout
        
        while True:
            check_count += 1
            
            # Emergency shutdown check
            if hasattr(self, 'coordinator') and hasattr(self.coordinator, '_emergency_shutdown'):
                if self.coordinator._emergency_shutdown:
                    return False
            
            # Send keep-alive command to prevent drone auto-landing
            if drone_instance and (time.time() - last_keepalive_time) >= KEEPALIVE_INTERVAL:
                try:
                    # Send RC stop command as keep-alive (ensures drone stays hovering)
                    drone_instance.send_rc_control(0, 0, 0, 0)
                    last_keepalive_time = time.time()
                    logger.log_debug('WaypointNavigationManager', 'Keep-alive sent during obstacle wait')
                except Exception as e:
                    logger.log_warning('WaypointNavigationManager', f'Keep-alive failed: {e}')
            
            # Get current frame
            try:
                frame = self.frame_read.frame
                if frame is None or frame.size == 0:
                    logger.log_warning('WaypointNavigationManager', 'No frame available for obstacle check')
                    time.sleep(self.OBSTACLE_CHECK_INTERVAL)
                    continue
            except Exception as e:
                logger.log_warning('WaypointNavigationManager', f'Frame read error: {e}')
                time.sleep(self.OBSTACLE_CHECK_INTERVAL)
                continue
            
            # Check for obstacles
            obstacle_detected, max_depth, annotated_frame = self.obstacle_detector.check_for_obstacles(frame)
            
            # Update latest depth frame for display
            if annotated_frame is not None:
                with self._depth_frame_lock:
                    self.latest_depth_frame = annotated_frame
            
            if not obstacle_detected:
                # Create a "PATH CLEAR" visual before moving forward
                if annotated_frame is not None:
                    clear_frame = annotated_frame.copy()
                    h, w = clear_frame.shape[:2]
                    
                    # Draw prominent green "PATH CLEAR" overlay
                    overlay = clear_frame.copy()
                    cv2.rectangle(overlay, (w//4, h//3), (3*w//4, 2*h//3), (0, 255, 0), -1)
                    cv2.addWeighted(overlay, 0.3, clear_frame, 0.7, 0, clear_frame)
                    
                    # Add "PATH CLEAR" text
                    cv2.putText(clear_frame, 'PATH CLEAR', (w//4 + 20, h//2), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.putText(clear_frame, 'Moving forward...', (w//4 + 30, h//2 + 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Update display with clear confirmation
                    with self._depth_frame_lock:
                        self.latest_depth_frame = clear_frame
                    
                    # Brief pause to show the "PATH CLEAR" message
                    time.sleep(0.3)
                
                if check_count > 1:
                    logger.log_success('WaypointNavigationManager', 
                        f'Path cleared after {check_count} checks ({time.time() - start_time:.1f}s)')
                return True
            
            # Obstacle detected - log and wait
            elapsed = time.time() - start_time
            logger.log_warning('WaypointNavigationManager', 
                f'Obstacle detected (P90: {max_depth:.0f}), waiting... '
                f'({elapsed:.1f}s / {self.OBSTACLE_CHECK_MAX_WAIT}s)')
            
            # Check timeout
            if elapsed >= self.OBSTACLE_CHECK_MAX_WAIT:
                logger.log_error('WaypointNavigationManager', 
                    f'Obstacle wait timeout after {self.OBSTACLE_CHECK_MAX_WAIT}s')
                return False
            
            time.sleep(self.OBSTACLE_CHECK_INTERVAL)
    
    def get_yaw(self, drone_instance=None) -> int:
        """Get current drone yaw angle from attitude telemetry."""
        try:
            attitude_str = drone_instance.send_command_with_return("attitude?", timeout=3)
            logger.log_debug('WaypointNavigationManager', f'Raw attitude: {attitude_str}')
            
            yaw = 0
            if attitude_str and ':' in attitude_str:
                parts = attitude_str.split(';')
                for part in parts:
                    if part.strip() and 'yaw:' in part:
                        try:
                            yaw_value = part.split(':')[1].strip()
                            if yaw_value:
                                yaw = int(yaw_value)
                        except:
                            continue
            return yaw
        except Exception as e:
            logger.log_warning('WaypointNavigationManager', f'Attitude query failed: {e}')
            return 0
    
    def get_current_waypoint_info(self) -> Tuple[str, str]:
        """Get current waypoint ID and display name."""
        if self.current_waypoint_id and self.current_waypoint_id in self.all_waypoints:
            waypoint = self.all_waypoints[self.current_waypoint_id]
            return waypoint.id, waypoint.name
        return "", ""
    
    def print_navigation_options(self):
        """Display formatted list of available navigation destinations."""
        destinations = self.get_available_destinations()
        current_id, current_name = self.get_current_waypoint_info()
        
        print(f"\n🏠 Current Position: {current_id} ('{current_name}')")
        
        # Group by type
        super_destinations = [d for d in destinations if d[2] == "super"]
        inner_destinations = [d for d in destinations if d[2] == "inner"]
        
        print("\n⭐ Super Waypoints:")
        print("-" * 40)
        if super_destinations:
            for wp_id, wp_name, wp_type in super_destinations:
                print(f"  • {wp_id}: '{wp_name}'")
        else:
            print("  (none available)")
        
        print("\n📍 Inner Waypoints:")
        print("-" * 40)
        if inner_destinations:
            for wp_id, wp_name, wp_type in inner_destinations:
                # Find parent Super Waypoint
                parent_id = self._get_parent_super_waypoint(wp_id)
                parent_name = self.super_waypoints[parent_id].name if parent_id else "?"
                print(f"  • {wp_id}: '{wp_name}' (in {parent_name})")
        else:
            print("  (none available)")
        
        return destinations
    
    # ==================== VIDEO DISPLAY METHODS ====================
    
    def start_video_display(self, window_name: str = "Navigation View"):
        """
        Start the video display thread showing camera feed and depth map.
        
        The display shows:
        - Top: Normal camera view
        - Bottom: MiDaS depth map with obstacle detection overlay
        
        Args:
            window_name: Name for the OpenCV display window
        """
        if self.video_display_running:
            logger.log_warning('WaypointNavigationManager', 'Video display already running')
            return
        
        self.video_display_running = True
        self._video_window_name = window_name
        self.video_display_thread = threading.Thread(
            target=self._video_display_loop, 
            daemon=True
        )
        self.video_display_thread.start()
        logger.log_info('WaypointNavigationManager', 'Video display thread started')
    
    def stop_video_display(self):
        """Stop the video display thread and close the window."""
        self.video_display_running = False
        if self.video_display_thread and self.video_display_thread.is_alive():
            self.video_display_thread.join(timeout=2)
        cv2.destroyAllWindows()
        logger.log_info('WaypointNavigationManager', 'Video display thread stopped')
    
    def pause_obstacle_detection(self):
        """
        Temporarily pause obstacle detection (MiDaS inference).
        
        Used during scanning operations when YOLO inference is running to prevent
        both models from running simultaneously (reduces GPU/CPU load).
        
        The video display will continue showing the camera feed but will stop
        updating the depth map.
        """
        self._obstacle_detection_paused = True
        logger.log_debug('WaypointNavigationManager', 'Obstacle detection paused (MiDaS inference suspended)')
    
    def resume_obstacle_detection(self):
        """
        Resume obstacle detection (MiDaS inference) after scanning completes.
        
        Re-enables depth map updates in the video display.
        """
        self._obstacle_detection_paused = False
        logger.log_debug('WaypointNavigationManager', 'Obstacle detection resumed (MiDaS inference active)')
    
    def set_external_frame_callback(self, callback):
        """
        Set an external callback to receive video frames for GUI display.
        
        When set, the video display loop will call this callback with frames
        instead of using cv2.imshow(). This allows the GUI to display the
        video feed without a separate OpenCV window.
        
        Args:
            callback: Function(camera_frame_bgr, depth_frame_bgr) to receive frames.
                      Both frames are BGR format, suitable for direct display or 
                      conversion to RGB for tkinter/PIL.
                      depth_frame_bgr may be None if no MiDaS model is loaded.
        """
        with self._external_frame_callback_lock:
            self._external_frame_callback = callback
        logger.log_debug('WaypointNavigationManager', 
            f'External frame callback {"set" if callback else "cleared"}')
    
    def _video_display_loop(self):
        """
        Video display loop running in separate thread.
        
        Shows split view with normal camera on top and depth map on bottom.
        Depth inference is throttled to reduce CPU/GPU load.
        """
        try:
            logger.log_debug('WaypointNavigationManager', 'Video display loop started')
            
            last_depth_update = 0
            DEPTH_UPDATE_INTERVAL = 0.2  # Update depth every 200ms (5 FPS) to reduce load
            cached_depth_frame = None
            
            while self.video_display_running:
                try:
                    if self.frame_read is None:
                        time.sleep(0.1)
                        continue
                    
                    # Get current camera frame
                    frame = self.frame_read.frame
                    if frame is None or frame.size == 0:
                        time.sleep(0.033)
                        continue
                    
                    # Resize camera frame to consistent width
                    display_width = 640
                    h, w = frame.shape[:2]
                    scale = display_width / w
                    display_height = int(h * scale)
                    
                    # Frame from PyAV/Tello is RGB, convert to BGR for cv2.imshow()
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_resized = cv2.resize(frame_bgr, (display_width, display_height))
                    
                    # Add header overlay to camera view (only camera title, no depth status)
                    overlay = frame_resized.copy()
                    cv2.rectangle(overlay, (0, 0), (display_width, 35), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, frame_resized, 0.3, 0, frame_resized)
                    cv2.putText(frame_resized, 'Navigation Camera View', 
                              (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Determine depth status for overlay on depth frame (simplified text)
                    if self._obstacle_detection_paused:
                        if self.obstacle_detector:
                            depth_status_text = 'Standby'
                            depth_status_color = (128, 128, 128)  # Gray - not actively running
                        else:
                            depth_status_text = 'No Model'
                            depth_status_color = (128, 128, 128)  # Gray
                    else:
                        depth_status_text = 'Active'
                        depth_status_color = (0, 255, 0)  # Green - actively running
                    
                    # Get depth map - use cached version or latest from obstacle checks
                    depth_display = None
                    current_time = time.time()
                    
                    # Skip MiDaS inference when paused (YOLO might be running)
                    if self._obstacle_detection_paused:
                        # Use cached depth frame if available, otherwise show paused placeholder
                        depth_display = cached_depth_frame
                    else:
                        # First try to get latest depth from obstacle check (already computed)
                        with self._depth_frame_lock:
                            if self.latest_depth_frame is not None:
                                depth_display = self.latest_depth_frame.copy()
                                cached_depth_frame = depth_display  # Cache it
                        
                        # If no recent depth frame, generate one with throttling
                        if depth_display is None and self.obstacle_detector is not None:
                            if (current_time - last_depth_update) >= DEPTH_UPDATE_INTERVAL:
                                try:
                                    _, _, annotated = self.obstacle_detector.check_for_obstacles(frame)
                                    if annotated is not None:
                                        depth_display = annotated
                                        cached_depth_frame = depth_display
                                        last_depth_update = current_time
                                except Exception as e:
                                    pass  # Silent fail, use cached
                            else:
                                # Use cached depth frame if available
                                depth_display = cached_depth_frame
                    
                    # Resize depth to match camera (or create placeholder)
                    if depth_display is not None:
                        depth_resized = cv2.resize(depth_display, (display_width, display_height))
                        # Add status overlay to depth map
                        overlay_depth = depth_resized.copy()
                        cv2.rectangle(overlay_depth, (0, 0), (display_width, 35), (0, 0, 0), -1)
                        cv2.addWeighted(overlay_depth, 0.7, depth_resized, 0.3, 0, depth_resized)
                        cv2.putText(depth_resized, f'Depth Map - {depth_status_text}', 
                                  (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, depth_status_color, 2)
                    else:
                        # No depth available - create placeholder
                        depth_resized = np.zeros((display_height, display_width, 3), dtype=np.uint8)
                        cv2.putText(depth_resized, 'Depth Map', (display_width//2 - 80, display_height//2 - 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
                        cv2.putText(depth_resized, '(No MiDaS model loaded)', (display_width//2 - 120, display_height//2 + 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
                    
                    # Check for external callback (GUI integration)
                    with self._external_frame_callback_lock:
                        external_callback = self._external_frame_callback
                    
                    if external_callback is not None:
                        # Send frames to external callback (GUI) instead of cv2.imshow
                        try:
                            external_callback(frame_resized, depth_resized)
                        except Exception as e:
                            logger.log_warning('WaypointNavigationManager', f'External callback error: {e}')
                    else:
                        # Use cv2.imshow for standalone display
                        combined = np.vstack((frame_resized, depth_resized))
                        cv2.imshow(self._video_window_name, combined)
                        cv2.waitKey(1)
                    
                    time.sleep(0.033)  # ~30 FPS
                    
                except Exception as e:
                    logger.log_warning('WaypointNavigationManager', f'Video display error: {e}')
                    time.sleep(0.1)
            
        except Exception as e:
            logger.log_error('WaypointNavigationManager', f'Video display thread error: {e}')
        finally:
            # Only destroy windows if we were using cv2.imshow
            with self._external_frame_callback_lock:
                if self._external_frame_callback is None:
                    cv2.destroyAllWindows()
            logger.log_debug('WaypointNavigationManager', 'Video display loop ended')

#!/usr/bin/env python3
"""
Linux-specific real-time drone controller for 2D hierarchical waypoint mapping.

Provides real-time manual control of Tello drones for creating 2D hierarchical waypoint maps
using Linux-compatible input handling (termios/select). Records all movements, tracks Super Waypoints
and inner waypoints, and generates JSON files for smart navigation.

Key Concepts:
- Super Waypoints: Hub waypoints that connect to other Super Waypoints
- Inner Waypoints: Local waypoints that branch off from Super Waypoints
- When marking inner waypoints, drone returns to associated Super Waypoint automatically
"""
import json
import os
import time
import uuid
import sys
import select
import termios
import tty
import threading
import traceback
import cv2
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

from dronebuddylib.utils.logger import Logger
from .video_grabber import TelloVideoGrabber

logger = Logger()


@dataclass
class Movement:
    """Represents a single drone movement record."""
    id: str
    type: str  # "move" or "lift"
    direction: Optional[str]  # For lift: "up" or "down"; For move: "forward", "backward", etc.
    distance: float
    start_yaw: int
    timestamp: str
    yaw: Optional[int] = None  # Calculated yaw for processed movements
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        if self.type == "move":
            return {
                'id': self.id,
                'type': self.type,
                'yaw': self.yaw,
                'distance': self.distance,
                'timestamp': self.timestamp
            }
        else:  # lift
            return {
                'id': self.id,
                'type': self.type,
                'direction': self.direction,
                'distance': self.distance,
                'timestamp': self.timestamp
            }
    
    def reverse(self) -> 'Movement':
        """Create a reversed movement for return navigation."""
        reversed_mov = Movement(
            id=str(uuid.uuid4()),
            type=self.type,
            direction=self._reverse_direction(),
            distance=self.distance,
            start_yaw=self.start_yaw,
            timestamp=datetime.now().isoformat(),
            yaw=self._reverse_yaw() if self.yaw is not None else None
        )
        return reversed_mov
    
    def _reverse_direction(self) -> Optional[str]:
        """Reverse vertical movement direction."""
        if self.type == "lift" and self.direction is not None:
            return "down" if self.direction == "up" else "up"
        return self.direction
    
    def _reverse_yaw(self) -> Optional[int]:
        """Calculate reverse yaw by adding 180 degrees."""
        if self.yaw is not None:
            reversed_raw = (self.yaw + 180)
            return reversed_raw if reversed_raw <= 180 else reversed_raw - 360
        return None


@dataclass
class InnerWaypoint:
    """Represents an inner waypoint within a Super Waypoint's local network."""
    id: str
    name: str
    movements_to_here: List[Movement] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'movements_to_here': [m.to_dict() for m in self.movements_to_here]
        }


@dataclass
class SuperWaypoint:
    """Represents a Super Waypoint hub with its inner waypoints."""
    id: str
    name: str
    index: int  # Position in super waypoint sequence
    movements_to_here: List[Movement] = field(default_factory=list)  # Movements from previous Super Waypoint
    inner_waypoints: List[InnerWaypoint] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'is_super_waypoint': True,
            'movements_to_here': [m.to_dict() for m in self.movements_to_here],
            'inner_waypoints': [iw.to_dict() for iw in self.inner_waypoints]
        }


class RealTimeDroneController:
    """Linux real-time controller for 2D hierarchical drone waypoint mapping."""
    
    def __init__(self, waypoint_dir: str, movement_speed: int, rotation_speed: int, nav_speed: int = 30, vertical_factor: float = 1.5):
        """Initialize controller with movement speeds and recording setup."""
        self.movement_speed = movement_speed  # cm/s for linear movements
        self.rotation_speed = rotation_speed  # degrees/s for rotations
        self.nav_speed = nav_speed  # Speed for automatic return movements
        self.vertical_factor = vertical_factor  # Compensation factor for vertical movements
        
        # 2D waypoint tracking
        self.super_waypoints: List[SuperWaypoint] = []  # All Super Waypoints
        self.current_super_waypoint: Optional[SuperWaypoint] = None  # Currently active Super Waypoint
        self.super_waypoint_counter = 0  # Sequential Super Waypoint numbering
        self.inner_waypoint_counter = 0  # Inner waypoint counter (resets per Super Waypoint)
        
        # Movement tracking state
        self.current_movement = None  # Active movement being recorded
        self.current_waypoint_movements: List[Movement] = []  # Movements since last waypoint marker
        
        # Control flags
        self.add_movement = False  # Whether to record current movement for waypoint data
        
        # Video streaming components
        self.video_thread = None
        self.video_running = False
        self.frame_read = None
        
        # JSON output file with timestamp
        filename = f"drone_movements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.data_file = os.path.join(waypoint_dir, filename)
        
        logger.log_info('RealTimeDroneController', f'Initialized 2D mapping controller')
    
    def get_drone_state(self, drone_instance=None) -> dict:
        """Get current drone telemetry data for movement recording."""
        try:
            state = {}

            # Get yaw (facing direction) from attitude command
            try:
                attitude_str = drone_instance.send_command_with_return("attitude?", timeout=3)
                logger.log_debug('RealTimeDroneController', f'Raw attitude response: {attitude_str}')
                
                # Parse attitude string format: "pitch:0;roll:0;yaw:45;"
                state['yaw'] = 0  # Default fallback value
                if attitude_str and ':' in attitude_str:
                    attitude_parts = attitude_str.split(';')
                    for part in attitude_parts:
                        if part.strip() and 'yaw:' in part:
                            try:
                                yaw_value = part.split(':')[1].strip()
                                if yaw_value:
                                    state['yaw'] = int(yaw_value)
                                    break
                            except (ValueError, IndexError) as e:
                                logger.log_warning('RealTimeDroneController', f'Failed to parse yaw from "{part}": {e}')
                                continue
            except Exception as e:
                logger.log_warning('RealTimeDroneController', f'Attitude query failed: {e}')
                state['yaw'] = 0

            # Get height in centimeters (converted from decimeters)
            try:
                height_str = drone_instance.send_command_with_return("height?", timeout=3)
                height_dm = int(height_str.replace('dm', ''))  # Remove 'dm' suffix
                state['height'] = height_dm * 10  # Convert dm to cm
            except Exception as e:
                logger.log_warning('RealTimeDroneController', f'Height query failed: {e}')
                state['height'] = 0
            
            # Get battery percentage
            try:
                battery_str = drone_instance.send_command_with_return("battery?", timeout=3)
                state['battery'] = int(battery_str)
            except Exception as e:
                logger.log_warning('RealTimeDroneController', f'Battery query failed: {e}')
                state['battery'] = 0

            return state
        except Exception as e:
            logger.log_error('RealTimeDroneController', f'Error getting drone state: {e}')
            return {'height': 0, 'yaw': 0, 'battery': 0}  # Return safe defaults on error
    
    def get_yaw(self, drone_instance=None) -> int:
        """Get current drone yaw angle from attitude telemetry."""
        try:
            attitude_str = drone_instance.send_command_with_return("attitude?", timeout=3)
            logger.log_debug('RealTimeDroneController', f'Raw attitude response: {attitude_str}')
            
            yaw = 0
            if attitude_str and ':' in attitude_str:
                attitude_parts = attitude_str.split(';')
                for part in attitude_parts:
                    if part.strip() and 'yaw:' in part:
                        try:
                            yaw_value = part.split(':')[1].strip()
                            if yaw_value:
                                yaw = int(yaw_value)
                        except (ValueError, IndexError) as e:
                            logger.log_warning('RealTimeDroneController', f'Failed to parse yaw: {e}')
                            continue
            return yaw
        except Exception as e:
            logger.log_warning('RealTimeDroneController', f'Attitude query failed: {e}')
            return 0

    def start_movement(self, direction: str, movement_type: str = "move", drone_instance=None):
        """Begin drone movement in specified direction and record movement data."""
        logger.log_debug('RealTimeDroneController', f'Starting movement: {movement_type} {direction}')
        
        if self.current_movement is not None:
            logger.log_warning('RealTimeDroneController', "Already moving, ignoring new movement")
            return
        
        # Get initial drone state
        try: 
            drone_state = self.get_drone_state(drone_instance)
            start_yaw = drone_state.get('yaw', 0)
        except Exception as e:
            logger.log_error('RealTimeDroneController', f'Error getting drone state: {e}')
            start_yaw = 0
        
        # Send RC control commands FIRST, then record timing
        try:
            if movement_type == "move":
                self.add_movement = True

                if direction == "forward":
                    drone_instance.send_rc_control(0, self.movement_speed, 0, 0)
                elif direction == "backward":
                    drone_instance.send_rc_control(0, -self.movement_speed, 0, 0)
                elif direction == "left":
                    drone_instance.send_rc_control(-self.movement_speed, 0, 0, 0)
                elif direction == "right":
                    drone_instance.send_rc_control(self.movement_speed, 0, 0, 0)

            elif movement_type == "lift":
                self.add_movement = True

                if direction == "up":
                    drone_instance.send_rc_control(0, 0, self.movement_speed, 0)
                elif direction == "down":
                    drone_instance.send_rc_control(0, 0, -self.movement_speed, 0)
                    
            elif movement_type == "rotate":
                self.add_movement = False  # Rotations don't create waypoint movements

                if direction == "anticlockwise":
                    drone_instance.send_rc_control(0, 0, 0, -self.rotation_speed)
                elif direction == "clockwise":
                    drone_instance.send_rc_control(0, 0, 0, self.rotation_speed)
            
            # Create movement record AFTER RC command is sent (timing starts when drone actually moves)
            self.current_movement = {
                'type': movement_type,
                'direction': direction,
                'start_time': time.time(),
                'start_yaw': start_yaw,
            }
            
            logger.log_debug('RealTimeDroneController', f'Created movement record: {self.current_movement}')
                    
        except Exception as e:
            logger.log_error('RealTimeDroneController', f'Error starting movement: {e}')
            traceback.print_exc()
            self.current_movement = None
    
    def stop_movement(self, drone_instance=None):
        """Stop current drone movement and record distance traveled."""
        if self.current_movement is None:
            return
        
        # Handle rotation movements separately
        if not self.add_movement:
            logger.log_debug('RealTimeDroneController', 'Stopping rotation movement...')
            try:
                drone_instance.send_rc_control(0, 0, 0, 0)
            except Exception as e:
                logger.log_error('RealTimeDroneController', f'Error stopping rotation: {e}')
            self.current_movement = None
            return
        
        # Stop movement
        try:
            drone_instance.send_rc_control(0, 0, 0, 0)
        except Exception as e:
            logger.log_error('RealTimeDroneController', f'Error stopping movement: {e}')
        
        # Calculate movement duration and distance
        end_time = time.time()
        duration = end_time - self.current_movement['start_time']
        distance = self.movement_speed * duration
        
        # Create Movement object
        movement = Movement(
            id=str(uuid.uuid4()),
            type=self.current_movement['type'],
            direction=self.current_movement['direction'],
            distance=round(distance, 2),
            start_yaw=self.current_movement['start_yaw'],
            timestamp=datetime.now().isoformat()
        )
        
        # Add to current movements list
        self.current_waypoint_movements.append(movement)

        logger.log_info('RealTimeDroneController', 
            f"Recorded {movement.type} {movement.direction} at {movement.start_yaw}°: {movement.distance:.1f}cm")
        
        self.current_movement = None
    
    def _process_movements(self, movements: List[Movement]) -> List[Movement]:
        """Process raw movements to calculate yaw values for horizontal movements."""
        processed = []
        
        for movement in movements:
            if movement.type == 'move':
                yaw = movement.start_yaw
                
                # Calculate yaw based on movement direction
                if movement.direction == 'forward':
                    yaw += 0
                elif movement.direction == 'backward':
                    yaw += 180
                elif movement.direction == 'left':
                    yaw -= 90
                elif movement.direction == 'right':
                    yaw += 90

                # Normalize to -180 to 180
                if yaw > 180: 
                    yaw -= 360
                elif yaw < -180:
                    yaw += 360
                
                movement.yaw = yaw
            
            processed.append(movement)
        
        return processed
    
    def _execute_return_to_super_waypoint(self, movements: List[Movement], drone_instance=None) -> bool:
        """Execute reverse movements to return drone to associated Super Waypoint."""
        if not movements:
            logger.log_info('RealTimeDroneController', 'No movements to reverse, already at Super Waypoint')
            return True
        
        logger.log_info('RealTimeDroneController', 
            f'Executing return to Super Waypoint with {len(movements)} reverse movements')
        
        print("\n🔄 RETURNING TO SUPER WAYPOINT...")
        print(f"Executing {len(movements)} reverse movements")
        
        # Reverse the movements (reverse order and reverse each movement)
        # Note: movements are already processed by caller
        reversed_movements = [mov.reverse() for mov in reversed(movements)]
        
        # Set navigation speed
        drone_instance.set_speed(self.nav_speed)
        
        try:
            for i, movement in enumerate(reversed_movements, 1):
                # Battery check before each movement
                try:
                    battery_str = drone_instance.send_command_with_return("battery?", timeout=3)
                    logger.log_debug('RealTimeDroneController', 'Checking battery status')
                    battery = int(battery_str)
                    if battery < 20:
                        logger.log_warning('RealTimeDroneController', f'Low battery detected: {battery}%')
                        if battery < 10:
                            logger.log_error('RealTimeDroneController', 
                                f'CRITICAL: Battery too low ({battery}%), aborting return')
                            return False
                except:
                    pass
                
                logger.log_debug('RealTimeDroneController', 
                    f'Return step {i}/{len(reversed_movements)}: {movement.type}')
                
                distance = max(movement.distance, 20)  # Minimum 20cm for drone commands
                
                if movement.type == "move":
                    # Handle horizontal movement with yaw orientation
                    yaw = movement.yaw if movement.yaw is not None else 0
                    current_yaw = self.get_yaw(drone_instance)
                    
                    # Calculate required yaw adjustment
                    turn_degree = abs(yaw - current_yaw)
                    if current_yaw > yaw:
                        if turn_degree > 180 and turn_degree < 360:
                            drone_instance.rotate_clockwise(360 - turn_degree)
                        elif turn_degree <= 180 and turn_degree > 0:
                            drone_instance.rotate_counter_clockwise(turn_degree)
                    else:
                        if turn_degree > 180 and turn_degree < 360:
                            drone_instance.rotate_counter_clockwise(360 - turn_degree)
                        elif turn_degree <= 180 and turn_degree > 0:
                            drone_instance.rotate_clockwise(turn_degree)
                    
                    drone_instance.send_rc_control(0, 0, 0, 0)
                    
                    # Execute forward movement
                    drone_instance.move_forward(int(distance))
                    drone_instance.send_rc_control(0, 0, 0, 0)
                    
                else:  # lift movement
                    if movement.direction == "up":
                        actual_distance = max(distance / self.vertical_factor, 20)
                        drone_instance.move_up(int(actual_distance))
                    elif movement.direction == "down":
                        drone_instance.move_down(int(distance))
                    
                    drone_instance.send_rc_control(0, 0, 0, 0)
                
                print(f"  ✓ Step {i}/{len(reversed_movements)} complete")
            
            logger.log_success('RealTimeDroneController', 'Successfully returned to Super Waypoint')
            print("✅ Returned to Super Waypoint!")
            return True
            
        except Exception as e:
            logger.log_error('RealTimeDroneController', f'Error during return: {e}')
            drone_instance.send_rc_control(0, 0, 0, 0)
            return False
    
    def mark_super_waypoint(self, name: str = None, auto_generated: bool = False) -> SuperWaypoint:
        """Mark current position as a Super Waypoint."""
        if not auto_generated and not name:
            name = input("Enter Super Waypoint name: ").strip()
            if not name:
                name = f"SuperWP_{self.super_waypoint_counter + 1}"
        
        self.super_waypoint_counter += 1
        self.inner_waypoint_counter = 0  # Reset inner counter for new Super Waypoint
        
        super_wp_id = f"SWP_{self.super_waypoint_counter:03d}"
        
        # Process movements to calculate yaw values
        processed_movements = self._process_movements(self.current_waypoint_movements.copy())
        
        # Create Super Waypoint
        super_wp = SuperWaypoint(
            id=super_wp_id,
            name=name or f"SuperWP_{self.super_waypoint_counter}",
            index=len(self.super_waypoints),
            movements_to_here=processed_movements,
            inner_waypoints=[]
        )
        
        self.super_waypoints.append(super_wp)
        self.current_super_waypoint = super_wp
        
        # Clear movements for next segment
        self.current_waypoint_movements = []
        
        logger.log_info('RealTimeDroneController', 
            f"Super Waypoint marked: {super_wp.name} (ID: {super_wp_id})")
        logger.log_info('RealTimeDroneController', 
            f"Movements from previous SWP: {len(processed_movements)}")
        
        print(f"\n⭐ SUPER WAYPOINT CREATED: '{super_wp.name}' ({super_wp_id})")
        
        return super_wp
    
    def mark_inner_waypoint(self, name: str = None, drone_instance=None) -> Optional[InnerWaypoint]:
        """Mark current position as an inner waypoint and return to Super Waypoint."""
        if self.current_super_waypoint is None:
            logger.log_error('RealTimeDroneController', 
                'Cannot mark inner waypoint - no Super Waypoint exists')
            return None
        
        if not name:
            name = input("Enter inner waypoint name: ").strip()
            if not name:
                self.inner_waypoint_counter += 1
                name = f"WP_{self.inner_waypoint_counter}"
        else:
            self.inner_waypoint_counter += 1
        
        # Create inner waypoint ID (format: SWP_001_IWP_001)
        inner_wp_id = f"{self.current_super_waypoint.id}_IWP_{self.inner_waypoint_counter:03d}"
        
        # Process movements
        processed_movements = self._process_movements(self.current_waypoint_movements.copy())
        
        # Create Inner Waypoint
        inner_wp = InnerWaypoint(
            id=inner_wp_id,
            name=name,
            movements_to_here=processed_movements
        )
        
        # Add to current Super Waypoint
        self.current_super_waypoint.inner_waypoints.append(inner_wp)
        
        logger.log_info('RealTimeDroneController', 
            f"Inner Waypoint marked: {inner_wp.name} (ID: {inner_wp_id})")
        logger.log_info('RealTimeDroneController', 
            f"Movements from Super Waypoint: {len(processed_movements)}")
        
        print(f"\n📍 INNER WAYPOINT CREATED: '{inner_wp.name}' ({inner_wp_id})")
        print(f"   Associated to: {self.current_super_waypoint.name} ({self.current_super_waypoint.id})")
        
        # Now return to Super Waypoint automatically
        print("\n🔄 Returning to associated Super Waypoint...")
        
        # Store movements before clearing (need them for return journey)
        movements_to_return = processed_movements.copy()
        
        # Clear movements for next segment
        self.current_waypoint_movements = []
        
        # Execute return to Super Waypoint
        success = self._execute_return_to_super_waypoint(movements_to_return, drone_instance)
        
        if success:
            print(f"✅ Back at Super Waypoint: {self.current_super_waypoint.name}")
        else:
            logger.log_warning('RealTimeDroneController', 
                'Return to Super Waypoint may not be complete')
        
        return inner_wp
    
    def mark_waypoint(self, name=None, auto_generated=False, drone_instance=None):
        """Interactive waypoint marking with Super Waypoint option."""
        if auto_generated:
            # Auto-generated waypoints are always Super Waypoints
            self.mark_super_waypoint(name, auto_generated=True)
            return
        
        print("\n" + "=" * 50)
        print("📍 MARK WAYPOINT")
        print("=" * 50)
        
        # Get waypoint name
        if not name:
            name = input("Enter waypoint name: ").strip()
            if not name:
                name = f"Waypoint_{self.super_waypoint_counter}_{self.inner_waypoint_counter + 1}"
        
        # Ask if this should be a Super Waypoint
        print("\n⭐ Make this a Super Waypoint?")
        print("   Super Waypoints are hub points that connect different areas.")
        print("   Inner waypoints are local points that branch off from Super Waypoints.")
        
        while True:
            choice = input("\nMake Super Waypoint? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                self.mark_super_waypoint(name)
                return
            elif choice in ['n', 'no']:
                self.mark_inner_waypoint(name, drone_instance)
                return
            else:
                print("Please enter 'y' or 'n'")
    
    def save_to_json(self) -> list:
        """Save 2D waypoint structure to JSON file."""
        # Build 2D waypoint data structure
        data = {
            'session_info': {
                'format_version': '2.0',
                'total_super_waypoints': len(self.super_waypoints),
                'total_inner_waypoints': sum(len(swp.inner_waypoints) for swp in self.super_waypoints),
                'created': datetime.now().isoformat()
            },
            'super_waypoints': [swp.to_dict() for swp in self.super_waypoints]
        }
        
        # Create summary for return value
        summary = []
        for swp in self.super_waypoints:
            summary.append({
                'id': swp.id, 
                'name': swp.name, 
                'type': 'super_waypoint',
                'inner_count': len(swp.inner_waypoints)
            })
            for iwp in swp.inner_waypoints:
                summary.append({
                    'id': iwp.id,
                    'name': iwp.name,
                    'type': 'inner_waypoint',
                    'parent_id': swp.id
                })
        
        # Save to file
        try:
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.log_success('RealTimeDroneController', f'Data saved to {self.data_file}')
            print(f"\n💾 Data saved to: {self.data_file}")
        except Exception as e:
            logger.log_error('RealTimeDroneController', f"Error saving data: {e}")
            summary = []
        
        return summary
    
    def start_video_stream(self, drone_instance=None):
        """Initialize and start video streaming from drone camera."""
        try:
            logger.log_info('RealTimeDroneController', 'Starting video stream...')
            
            drone_instance.streamon()
            time.sleep(3)
            
            self.frame_read = drone_instance.get_frame_read()
            
            retry_count = 0
            max_retries = 10
            while retry_count < max_retries:
                try:
                    test_frame = self.frame_read.frame
                    if test_frame is not None and test_frame.size > 0:
                        break
                except:
                    pass
                retry_count += 1
                time.sleep(0.5)
                logger.log_debug('RealTimeDroneController', 
                    f'Waiting for video stream... ({retry_count}/{max_retries})')
            
            if retry_count >= max_retries:
                raise Exception("Video stream failed to initialize")
            
            self.video_running = True
            self.video_thread = threading.Thread(target=self._video_display_loop, daemon=True)
            self.video_thread.start()
            
            logger.log_success('RealTimeDroneController', 'Video stream started successfully.')
            logger.log_info('RealTimeDroneController', 'Video stream window opened - you can see what the drone sees!')
            logger.log_info('RealTimeDroneController', "Keep the video window visible to see the drone's perspective during mapping.")

        except Exception as e:
            logger.log_error('RealTimeDroneController', 'Failed to initialize video stream, continuing mapping without video feed.')
            self.stop_video_stream(drone_instance=drone_instance)

    def stop_video_stream(self, drone_instance=None):
        """Stop video streaming and clean up resources."""
        try:
            logger.log_info('RealTimeDroneController', 'Stopping video stream...')
            
            self.video_running = False
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=2)

            self.frame_read = None
            cv2.destroyAllWindows()
            
            if drone_instance:
                drone_instance.streamoff()
            
            logger.log_success('RealTimeDroneController', 'Video stream stopped.')
            
        except Exception as e:
            logger.log_error('RealTimeDroneController', f'Error stopping video stream: {e}')
    
    def _video_display_loop(self):
        """Background thread for continuous video display with overlay information."""
        try:
            window_name = 'Drone Camera - Mapping Mode (Linux)'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            logger.log_debug('RealTimeDroneController', 'Video display thread started.')
            
            while self.video_running and self.frame_read:
                try:
                    frame = self.frame_read.frame
                    
                    if frame is not None and frame.size > 0:
                        # Resize frame if too large for display
                        height, width = frame.shape[:2]
                        if width > 960:
                            scale = 960 / width
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            frame = cv2.resize(frame, (new_width, new_height))
                        
                        # Create overlay for UI elements with transparency
                        overlay = frame.copy()
                        
                        # Draw header background (black with transparency)
                        height, width = frame.shape[:2]
                        cv2.rectangle(overlay, (0, 0), (width, 110), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                        
                        # Add title and control instructions
                        cv2.putText(frame, 'Drone Camera View - Mapping Mode (Linux)', 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, 'Use terminal for controls - Press Q in terminal to quit', 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Show current Super Waypoint info
                        if self.current_super_waypoint:
                            swp_text = f"SWP: {self.current_super_waypoint.name} | Inner: {len(self.current_super_waypoint.inner_waypoints)}"
                            cv2.putText(frame, swp_text, 
                                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                        cv2.imshow(window_name, frame)
                        cv2.waitKey(1)
                    
                    time.sleep(0.033)  # Approximately 30 FPS
                    
                except Exception as e:
                    logger.log_warning('RealTimeDroneController', f'Frame display error: {e}')
                    time.sleep(0.1)
            
            logger.log_debug('RealTimeDroneController', 'Video display thread ended.')
            
        finally:
            cv2.destroyAllWindows()
    
    def get_key(self):
        """Get single keypress with timeout - Linux implementation."""
        if select.select([sys.stdin], [], [], 0.2) == ([sys.stdin], [], []):
            key = sys.stdin.read(1).lower()
            
            if key == '\x1b':  # Escape sequence (arrow keys start with ESC)

                time.sleep(0.02)  # Allow time for complete escape sequence
                if select.select([sys.stdin], [], [], 0.1)[0]: 
                    bracket = sys.stdin.read(1)
                    if bracket == '[' and select.select([sys.stdin], [], [], 0.1)[0]:
                        arrow = sys.stdin.read(1)
                        arrow_map = {
                            'A': 'up',  # Up arrow
                            'B': 'down',  # Down arrow
                            'C': 'right',  # Right arrow
                            'D': 'left'   # Left arrow
                        }
                        return arrow_map.get(arrow, 'unknown_key')
                return 'incomplete'
            elif key == '[': 
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    sys.stdin.read(1)
                return 'ignored_key'  
            else: 
                return key
        return None

    def handle_keypress(self, drone_instance=None):
        """Main keyboard input handler for drone control."""
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            tty.setraw(sys.stdin)
            
            activeMovementKey = None
            x_pressed = False
            last_battery_check = 0
            
            print("\n🎮 2D Mapping Controls Active!")
            print("=" * 50)
            print("Movement: W/A/S/D (horizontal), Up/Down arrows (vertical)")
            print("Rotation: Left/Right arrows")
            print("X = Mark waypoint")
            print("Q = Finish mapping")
            print("=" * 50)
            
            while True:
                # Battery monitoring
                current_time = time.time()
                if current_time - last_battery_check > 5:
                    try:
                        battery_str = drone_instance.send_command_with_return("battery?", timeout=5)
                        battery = int(battery_str)
                        if battery < 20:
                            logger.log_warning('RealTimeDroneController', f'Low battery: {battery}%')
                            if battery < 10:
                                logger.log_error('RealTimeDroneController', 'CRITICAL: Battery too low')
                                break
                        last_battery_check = current_time
                    except Exception as e:
                        logger.log_error('RealTimeDroneController', f'Battery check error: {e}')

                key = self.get_key()
                
                if key:
                    logger.log_debug('RealTimeDroneController', f'Key: {key}')

                    if key == 'q':
                        logger.log_info('RealTimeDroneController', 'Finishing mapping session...')
                        break
                        
                    elif key == 'x':
                        if not x_pressed:
                            if self.current_movement:
                                self.stop_movement(drone_instance=drone_instance)
                                activeMovementKey = None

                            logger.log_info('RealTimeDroneController', 'Marking waypoint...')
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                            self.mark_waypoint(drone_instance=drone_instance)
                            tty.setraw(sys.stdin)
                            x_pressed = True
                        else:
                            logger.log_info('RealTimeDroneController', 'Waypoint already marked')
                            continue
                            
                    elif key in ['w', 'a', 's', 'd', 'up', 'down', 'left', 'right']:
                        x_pressed = False
                        if key != activeMovementKey:
                            # Stop current movement if any
                            if activeMovementKey:
                                self.stop_movement(drone_instance=drone_instance)

                            # Start new movement
                            activeMovementKey = key

                            if key == 'w':
                                self.start_movement('forward', 'move', drone_instance)
                            elif key == 's':
                                self.start_movement('backward', 'move', drone_instance)
                            elif key == 'a':
                                self.start_movement('left', 'move', drone_instance)
                            elif key == 'd':
                                self.start_movement('right', 'move', drone_instance)
                            elif key == 'up':
                                self.start_movement('up', 'lift', drone_instance)
                            elif key == 'down':
                                self.start_movement('down', 'lift', drone_instance)
                            elif key == 'left':
                                self.start_movement('anticlockwise', 'rotate', drone_instance)
                            elif key == 'right':
                                self.start_movement('clockwise', 'rotate', drone_instance)
                        else:
                            # Same movement key held - continue current movement
                            continue
                        continue
                    else:
                        if self.current_movement:
                            self.stop_movement(drone_instance=drone_instance)
                            activeMovementKey = None
                        continue
                else:
                    if self.current_movement:
                        self.stop_movement(drone_instance=drone_instance)
                        activeMovementKey = None
                    continue
                
                time.sleep(0.05)  # Fast response loop (50ms cycle time)
                
        except Exception as e:
            logger.log_error('RealTimeDroneController', f'Error in keyboard handling: {e}')
        finally:
            # Always restore terminal settings on exit
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            logger.log_info('RealTimeDroneController', 'Keyboard controls ended')

    
    def run(self, drone_instance=None) -> list:
        """Main entry point - initialize 2D hierarchical mapping session and handle complete workflow."""
        
        print("\n" + "=" * 60)
        print("🗺️  2D HIERARCHICAL WAYPOINT MAPPING")
        print("=" * 60)
        print("\nThis system creates a web of interconnected waypoints:")
        print("- Super Waypoints: Hub points that connect different areas")
        print("- Inner Waypoints: Local points that branch off from hubs")
        print("\nThe drone automatically returns to Super Waypoints after")
        print("marking inner waypoints, creating a navigable network.")
        print("=" * 60)
        
        # Mark the first waypoint as a Super Waypoint automatically
        self.mark_super_waypoint("START", auto_generated=True)
        logger.log_info('RealTimeDroneController', 'First Super Waypoint marked: START')

        # Start video feed for visual reference during mapping
        self.start_video_stream(drone_instance=drone_instance)

        try:
            self.handle_keypress(drone_instance=drone_instance)
        except KeyboardInterrupt:
            logger.log_info('RealTimeDroneController', 'Keyboard interrupt received')
        except Exception as e:
            logger.log_error('RealTimeDroneController', f'Error during drone control: {e}')
        finally:
            # Stop video streaming
            self.stop_video_stream(drone_instance=drone_instance)
            
            # Complete any pending movements and save session data
            try: 
                if self.current_movement:
                    self.stop_movement(drone_instance=drone_instance)
                
                # Handle final position if there are pending movements
                if self.current_waypoint_movements:
                    print("\n📍 Marking final position as END waypoint...")
                    
                    # Mark as inner waypoint (will auto-return to Super Waypoint)
                    inner_wp = InnerWaypoint(
                        id=f"{self.current_super_waypoint.id}_IWP_END",
                        name="END",
                        movements_to_here=self._process_movements(self.current_waypoint_movements.copy())
                    )
                    self.current_super_waypoint.inner_waypoints.append(inner_wp)
                    
                    # Return to Super Waypoint before landing
                    print("\n🔄 Returning to Super Waypoint before landing...")
                    self._execute_return_to_super_waypoint(
                        self._process_movements(self.current_waypoint_movements.copy()),
                        drone_instance
                    )
                    
                    self.current_waypoint_movements = []

                # Save complete mapping session to JSON
                summary = self.save_to_json()

                # Print summary
                print("\n" + "=" * 60)
                print("📊 MAPPING SESSION SUMMARY")
                print("=" * 60)
                print(f"Total Super Waypoints: {len(self.super_waypoints)}")
                total_inner = sum(len(swp.inner_waypoints) for swp in self.super_waypoints)
                print(f"Total Inner Waypoints: {total_inner}")
                print(f"Data saved to: {self.data_file}")
                print("=" * 60)
                
            except Exception as e:
                logger.log_error('RealTimeDroneController', f'Error finalizing session: {e}')
                summary = []

            return summary  # Return list of waypoint summaries

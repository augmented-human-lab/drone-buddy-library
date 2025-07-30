#!/usr/bin/env python3
"""
Linux-specific real-time drone controller for waypoint mapping.

Provides real-time manual control of Tello drones for creating waypoint maps
using Linux-compatible input handling (termios/select). Records all movements,
tracks waypoints, and generates JSON files for navigation.
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
from datetime import datetime

from dronebuddylib.utils.logger import Logger
from .video_grabber import TelloVideoGrabber

logger = Logger()


class RealTimeDroneController:
    """Linux real-time controller for drone waypoint mapping through manual flight."""
    
    def __init__(self, waypoint_dir: str, movement_speed: int, rotation_speed: int):
        """Initialize controller with movement speeds and recording setup."""
        self.movement_speed = movement_speed  # cm/s for linear movements
        self.rotation_speed = rotation_speed  # degrees/s for rotations
        
        # Movement tracking state
        self.current_movement = None  # Active movement being recorded
        self.waypoints = []  # Complete waypoint list
        self.current_waypoint_movements = []  # Movements since last waypoint
        self.waypoint_counter = 0  # Sequential waypoint numbering
        
        # Control flags
        self.add_movement = False  # Whether to record current movement
        
        # Video streaming components
        self.video_thread = None
        self.video_running = False
        self.frame_read = None
        
        # JSON output file with timestamp
        filename = f"drone_movements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.data_file = os.path.join(waypoint_dir, filename)
    
    def get_drone_state(self, drone_instance=None):
        """Get current drone telemetry data for movement recording."""
        try:
            state = {}

            # Get yaw (facing direction) from attitude command
            try:
                attitude_str = drone_instance.send_command_with_return("attitude?", timeout=3)
                logger.log_debug('RealTimeDroneController', f'Raw attitude response: {attitude_str}')
                
                # Parse attitude string format: "pitch:0;roll:0;yaw:45;"
                state['yaw'] = 0  # Default value
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
                height_str = self.tello.send_command_with_return("height?", timeout=3)
                height_dm = int(height_str.replace('dm', ''))  # Remove 'dm' suffix
                state['height'] = height_dm * 10  # Convert dm to cm
            except Exception as e:
                logger.log_warning('RealTimeDroneController', f'Height query failed: {e}')
                state['height'] = 0
            
            # Get battery percentage
            try:
                battery_str = self.tello.send_command_with_return("battery?", timeout=3)
                state['battery'] = int(battery_str)
            except Exception as e:
                logger.log_warning('RealTimeDroneController', f'Battery query failed: {e}')
                state['battery'] = 0
            

            return state
        except Exception as e:
            logger.log_error('RealTimeDroneController', f'Error getting drone state: {e}')
            return {'height': 0, 'yaw': 0, 'battery': 0}  # Safe defaults

    def start_movement(self, direction, movement_type="move", drone_instance=None):
        """Begin drone movement in specified direction and record movement data."""
        logger.log_debug('RealTimeDroneController', f'Starting movement: {movement_type} {direction}')
        
        if self.current_movement is not None:
            logger.log_warning('RealTimeDroneController', "Already moving, ignoring new movement")
            return  # Prevent overlapping movements
        
        # Get initial drone state for movement recording
        try: 
            drone_state = self.get_drone_state(drone_instance)
            start_yaw = drone_state.get('yaw', 0)  # Default to 0 if unavailable
        except Exception as e:
            logger.log_error('RealTimeDroneController', f'Error getting drone state: {e}')
            start_yaw = 0

        # Create movement record with timing and orientation
        self.current_movement = {
            'type': movement_type,
            'direction': direction,
            'start_time': time.time(),
            'start_yaw': start_yaw,
        }

        logger.log_debug('RealTimeDroneController', f'Created movement record: {self.current_movement}')

        # Send appropriate RC control commands to drone
        try:
            if movement_type == "move":
                self.add_movement = True  # Linear movements are recorded

                logger.log_debug('RealTimeDroneController', f'Sending RC control for {direction}')
                if direction == "forward":
                    drone_instance.send_rc_control(0, self.movement_speed, 0, 0)
                elif direction == "backward":
                    drone_instance.send_rc_control(0, -self.movement_speed, 0, 0)
                elif direction == "left":
                    drone_instance.send_rc_control(-self.movement_speed, 0, 0, 0)
                elif direction == "right":
                    drone_instance.send_rc_control(self.movement_speed, 0, 0, 0)
                logger.log_debug('RealTimeDroneController', f'RC control sent for {direction}')

            elif movement_type == "lift":
                self.add_movement = True  # Vertical movements are recorded

                logger.log_debug('RealTimeDroneController', f'Sending RC control for lift {direction}')
                if direction == "up":
                    drone_instance.send_rc_control(0, 0, self.movement_speed, 0)
                elif direction == "down":
                    drone_instance.send_rc_control(0, 0, -self.movement_speed, 0)
                logger.log_debug('RealTimeDroneController', f'RC control sent for lift {direction}')
                    
            elif movement_type == "rotate":
                self.add_movement = False  # Rotations are not recorded as waypoint movements

                logger.log_debug('RealTimeDroneController', f'Sending RC control for rotate {direction}')
                if direction == "anticlockwise":
                    drone_instance.send_rc_control(0, 0, 0, -self.rotation_speed)
                elif direction == "clockwise":
                    drone_instance.send_rc_control(0, 0, 0, self.rotation_speed)
                logger.log_debug('RealTimeDroneController', f'RC control sent for rotate {direction}')
                    
        except Exception as e:
            logger.log_error('RealTimeDroneController', f'Error starting movement: {e}')
            import traceback
            traceback.print_exc()
            self.current_movement = None
    
    def stop_movement(self, drone_instance=None):
        """Stop current movement, calculate distance, and record movement event."""
        if self.current_movement is None:
            return
        
        if not self.add_movement:
            # Handle rotation movements (not recorded as waypoint data)
            logger.log_debug('RealTimeDroneController', 'Stopping rotation movement...')
            try:
                drone_instance.send_rc_control(0, 0, 0, 0)  # Stop all RC control
            except Exception as e:
                logger.log_error('RealTimeDroneController', f'Error stopping rotation movement: {e}')

            self.current_movement = None
            return
        
        # Stop drone movement
        try:
            drone_instance.send_rc_control(0, 0, 0, 0)  # Stop all movement
        except Exception as e:
            logger.log_error('RealTimeDroneController', f'Error stopping movement: {e}')

        # Calculate movement duration and distance for recording
        end_time = time.time()
        duration = end_time - self.current_movement['start_time'] + 0.5  # Add buffer for deceleration
        
        distance = self.movement_speed * duration  # cm
        
        # Create movement event record for waypoint file
        movement_event = {
            'id': str(uuid.uuid4()),
            'type': self.current_movement['type'],
            'direction': self.current_movement['direction'],
            'distance': round(distance, 2),
            'start_yaw': self.current_movement['start_yaw'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to current waypoint's movement list
        self.current_waypoint_movements.append(movement_event)

        logger.log_info('RealTimeDroneController', f"Recorded {movement_event['type']} {movement_event['direction']} at {movement_event['start_yaw']} degree(s): "
              f"{movement_event['distance']:.1f}cm")
        
        self.current_movement = None
    
    def mark_waypoint(self, name=None, auto_generated=False):
        """Create a new waypoint with current movement cluster and reset for next waypoint."""
        if not auto_generated and not name:
            name = input("Enter waypoint name: ").strip()
            if not name:
                name = f"Waypoint_{self.waypoint_counter + 1}"
        
        self.waypoint_counter += 1
        waypoint_id = f"WP_{self.waypoint_counter:03d}"  # Format: WP_001, WP_002, etc.
        
        waypoint = {
            'id': waypoint_id,
            'name': name or f"Waypoint_{self.waypoint_counter}",
            'movements_to_here': self.current_waypoint_movements.copy()
        }
        
        self.waypoints.append(waypoint)
        
        logger.log_info('RealTimeDroneController', f"Waypoint marked: {waypoint['name']} (ID: {waypoint_id})")
        logger.log_info('RealTimeDroneController', f"Movements recorded: {len(self.current_waypoint_movements)} events")

        # Reset movement list for next waypoint cluster
        self.current_waypoint_movements = []
    
    def save_to_json(self) -> list:
        """Process movement data and save complete waypoint map to JSON file."""
        processed_waypoints = []
        for waypoint in self.waypoints: 
            processed_movements = []

            # Process movements - convert directions to yaw angles for 'move' type
            for movement in waypoint['movements_to_here']:
                if movement['type'] == 'move':
                    yaw = movement['start_yaw']
                    # Adjust yaw based on movement direction
                    if movement['direction'] == 'forward':
                        yaw += 0
                    elif movement['direction'] == 'backward':
                        yaw += 180
                    elif movement['direction'] == 'left':
                        yaw -= 90
                    elif movement['direction'] == 'right':
                        yaw += 90
                    
                    # Normalize yaw to -180 to 180 range
                    if yaw > 180: 
                        yaw -= 360
                    elif yaw < -180:
                        yaw += 360
                    
                    processed_movement = {
                        'id': movement['id'],
                        'type': movement['type'],
                        'yaw': yaw, 
                        'distance': movement['distance'],
                        'timestamp': movement['timestamp']
                    }

                    processed_movements.append(processed_movement)
                
                else: 
                    # For 'lift' movements, keep direction and distance
                    processed_movement = {
                        'id': movement['id'],
                        'type': movement['type'],
                        'direction': movement['direction'],
                        'distance': movement['distance'],
                        'timestamp': movement['timestamp']
                    }
                    processed_movements.append(processed_movement)

            processed_waypoint = {
                'id': waypoint['id'],
                'name': waypoint['name'],
                'movements_to_here': processed_movements
            }

            processed_waypoints.append(processed_waypoint)

        # Create final JSON structure
        data = {
            'session_info': {
                'total_waypoints': len(self.waypoints),
            },
            'waypoints': processed_waypoints
        }
        
        summary = [{'id': wp['id'], 'name': wp['name']} for wp in processed_waypoints]

        try:
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.log_success('RealTimeDroneController', f'Data saved to {self.data_file}')
        except Exception as e:
            logger.log_error('RealTimeDroneController', f"Error saving data: {e}")
            summary = []
        finally: 
            return summary  # Return summary of waypoint IDs and names
    
    def start_video_stream(self, drone_instance=None):
        """Start video streaming from drone camera using Tello's built-in function."""
        try:
            logger.log_info('RealTimeDroneController', 'Starting video stream...')
            
            # Start video stream
            drone_instance.streamon()
            time.sleep(3)  # Wait a bit longer for stream to initialize properly
            
            # Get frame reader using Tello's built-in function
            self.frame_read = drone_instance.get_frame_read()
            
            # Wait for first frame to be available
            retry_count = 0
            max_retries = 10
            while retry_count < max_retries:
                try:
                    test_frame = self.frame_read.frame
                    if test_frame is not None and test_frame.size > 0:
                        break
                except:
                    pass  # Ignore errors, just retry
                retry_count += 1
                time.sleep(0.5)
                logger.log_debug('RealTimeDroneController', f'Waiting for video stream... ({retry_count}/{max_retries})')
            
            if retry_count >= max_retries:
                raise Exception("Video stream failed to initialize - no frames received")
            
            # Start video display thread
            self.video_running = True
            self.video_thread = threading.Thread(target=self._video_display_loop, daemon=True)
            self.video_thread.start()
            
            logger.log_success('RealTimeDroneController', 'Video stream started successfully.')
            logger.log_info('RealTimeDroneController', 'Video stream window opened - you can see what the drone sees!')
            logger.log_info('RealTimeDroneController', "Keep the video window visible to see the drone's perspective during mapping.")
            
        except Exception as e:
            logger.log_error('RealTimeDroneControllerWindows', 'Failed to initialize video stream, continuing mapping without video feed.')

            self.stop_video_stream(drone_instance=drone_instance)

    def stop_video_stream(self, drone_instance=None):
        """Stop video streaming."""
        try:
            logger.log_info('RealTimeDroneController', 'Stopping video stream...')
            
            # Stop video thread
            self.video_running = False
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=2)

            self.frame_read = None
            
            # Close OpenCV windows
            cv2.destroyAllWindows()
            
            # Stop drone video stream
            if drone_instance:
                drone_instance.streamoff()
            
            logger.log_success('RealTimeDroneController', 'Video stream stopped.')
            
        except Exception as e:
            logger.log_error('RealTimeDroneController', f'Error stopping video stream: {e}')

    def _video_display_loop(self):
        """Video display loop running in separate thread."""
        try:
            logger.log_debug('RealTimeDroneController', 'Video display thread started.')
            
            while self.video_running and self.frame_read:
                try:
                    # Get current frame
                    frame = self.frame_read.frame
                    
                    if frame is not None and frame.size > 0:
                        # Resize frame 
                        height, width = frame.shape[:2]
                        if width > 960:  # Resize if too large
                            scale = 960 / width
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            frame = cv2.resize(frame, (new_width, new_height))
                        
                        # Add overlay text with background for better visibility
                        overlay = frame.copy()
                        
                        # Header background
                        cv2.rectangle(overlay, (0, 0), (width, 110), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                        
                        # Title and instructions
                        cv2.putText(frame, 'Drone Camera View - Mapping Mode (Windows)', 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, 'Use terminal for controls - Press Q in terminal to quit', 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Display frame
                        cv2.imshow('Drone Camera - Mapping Mode (Windows)', frame)
                        
                        # Handle window events (but don't wait for key presses)
                        cv2.waitKey(1)
                    
                    # Small delay to prevent excessive CPU usage
                    time.sleep(0.033)  # ~30 FPS
                    
                except Exception as e:
                    logger.log_warning('RealTimeDroneController', f'Frame display error: {e}')
                    time.sleep(0.1)  # Wait before retry
            
            logger.log_debug('RealTimeDroneController', 'Video display thread ended.')
            
        finally:
            # Ensure window is closed
            cv2.destroyAllWindows()
    
    def get_key(self):
        """Linux keyboard input with arrow key detection using select and termios."""
        if select.select([sys.stdin], [], [], 0.5) == ([sys.stdin], [], []):
            # Read single character from stdin
            key = sys.stdin.read(1).lower()
            
            if key == '\x1b':  # Escape sequence (arrow keys start with ESC)

                time.sleep(0.02)  # Allow time for escape sequence
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
                # Ignore alphabet character that follows
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    sys.stdin.read(1)
                return 'ignored_key'  
            else: 
                return key  # Regular key press
        return None

    def handle_keypress(self, drone_instance=None):
        """Main keyboard control loop using Linux termios for real-time input."""
        
        # Save original terminal settings for restoration
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            # Set terminal to raw mode for immediate key detection
            tty.setraw(sys.stdin)
            
            activeMovementKey = None
            x_pressed = False
            last_battery_check = 0
            
            print("🎮 Keyboard controls active!")
            
            while True:
                # Battery monitoring every 5 seconds
                current_time = time.time()
                if current_time - last_battery_check > 5:
                    try:
                        battery_str = drone_instance.send_command_with_return("battery?", timeout=5)
                        battery = int(battery_str)
                        if battery < 20:
                            logger.log_warning('RealTimeDroneController', f'Low battery ({battery}%)')
                            if battery < 10:
                                logger.log_error('RealTimeDroneController', 'CRITICAL: Battery too low, landing...')
                                break
                        last_battery_check = current_time
                    except Exception as e:
                        logger.log_error('RealTimeDroneController', f'Error checking battery: {e}')

                # Get keyboard input
                key = self.get_key()
                
                if key:
                    logger.log_info('RealTimeDroneController', f'Key: {key}')

                    if key == 'q':
                        logger.log_info('RealTimeDroneController', 'Finishing mapping session')
                        break
                    elif key == 'x': 
                        if not x_pressed:
                            if self.current_movement:
                                self.stop_movement(drone_instance=drone_instance)
                                activeMovementKey = None

                            logger.log_info('RealTimeDroneController', 'Marking waypoint...')
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)  # Restore for input

                            self.mark_waypoint()

                            old_settings = termios.tcgetattr(sys.stdin)  # Save again
                            tty.setraw(sys.stdin)  # Return to raw mode
                            x_pressed = True
                        else:
                            logger.log_info('RealTimeDroneController', 'Waypoint already marked')
                            continue
                    elif key in ['w', 'a', 's', 'd', 'up', 'down', 'left', 'right']:
                        x_pressed = False  # Reset waypoint flag
                        if key != activeMovementKey:
                            # Stop current movement before starting new one
                            if activeMovementKey:
                                logger.log_info('RealTimeDroneController', f'Stopping movement: {activeMovementKey}')
                                self.stop_movement(drone_instance=drone_instance)

                            # Start new movement
                            activeMovementKey = key
                            logger.log_info('RealTimeDroneController', f'Starting movement: {key}')
                            
                            # Map keys to movements using match-case (Python 3.10+)
                            match key:
                                case 'w': 
                                    self.start_movement('forward', 'move', drone_instance)
                                case 's': 
                                    self.start_movement('backward', 'move', drone_instance)
                                case 'a': 
                                    self.start_movement('left', 'move', drone_instance)
                                case 'd': 
                                    self.start_movement('right', 'move', drone_instance)
                                case 'up':  
                                    self.start_movement('up', 'lift', drone_instance)
                                case 'down':  
                                    self.start_movement('down', 'lift', drone_instance)
                                case 'left':  
                                    self.start_movement('anticlockwise', 'rotate', drone_instance)
                                case 'right':  
                                    self.start_movement('clockwise', 'rotate', drone_instance)
                        else: 
                            # Same movement continues
                            logger.log_info('RealTimeDroneController', f'Continuing movement: {activeMovementKey}')
                            continue
                    else:
                        # Unrecognized key - stop movement for safety
                        logger.log_info('RealTimeDroneController', f'Unrecognized key: {key}')
                        if self.current_movement:
                            logger.log_info('RealTimeDroneController', 'Stopping current movement due to unrecognized key')
                            self.stop_movement(drone_instance=drone_instance)
                            activeMovementKey = None
                        continue
                else:
                    # No key pressed - stop any active movement
                    if self.current_movement:
                        logger.log_info('RealTimeDroneController', 'No key pressed, stopping current movement')
                        self.stop_movement(drone_instance=drone_instance)
                        activeMovementKey = None
                    continue
                
                time.sleep(0.05)  # Fast responsive loop (50ms)
                
        except Exception as e:
            logger.log_error('RealTimeDroneController', f'Error in keyboard handling: {e}')
        finally:
            # Always restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            logger.log_info('RealTimeDroneController', 'Keyboard controls ended')

    
    def run(self, drone_instance=None) -> list:
        """Main entry point - initialize mapping session and handle complete workflow."""
        
        print("Starting keyboard control... Press Q to exit")
        
        # Auto-create START waypoint
        self.mark_waypoint("START", auto_generated=True)
        logger.log_info('RealTimeDroneController', 'First waypoint marked: START')

        # Start video feed for visual reference
        self.start_video_stream(drone_instance=drone_instance)

        try:
            self.handle_keypress(drone_instance=drone_instance)
        except KeyboardInterrupt:
            logger.log_info('RealTimeDroneController', 'Keyboard interrupt received')
        except Exception as e:
            logger.log_error('RealTimeDroneController', f'Error during drone control: {e}')
        finally:
            # Cleanup and finalization
            self.stop_video_stream(drone_instance=drone_instance)
            
            # Complete any pending movements and save session
            try: 
                if self.current_movement:
                    self.stop_movement(drone_instance=drone_instance)
                
                # Auto-create END waypoint if movements exist
                if self.current_waypoint_movements:
                    self.mark_waypoint("END", auto_generated=True)

                # Save complete mapping session to JSON
                summary = self.save_to_json()

                print(f"\nSession complete! Data saved to: {self.data_file}")
            except Exception as e:
                logger.log_error('RealTimeDroneController', f'Error finalizing session: {e}')
                summary = []

            return summary  # Return list of waypoint summaries

#!/usr/bin/env python3
"""
Windows-specific real-time drone controller for waypoint mapping.

Provides real-time manual control of Tello drones for creating waypoint maps
using Windows-compatible input handling (msvcrt). Records all movements,
tracks waypoints, and generates JSON files for navigation.
"""
import json
import os
import time
import uuid
import threading
import cv2
import msvcrt
import win32gui
import win32con
from datetime import datetime

from dronebuddylib.utils.logger import Logger

logger = Logger()


class RealTimeDroneControllerWindows:
    """Windows real-time controller for drone waypoint mapping through manual flight."""
    
    def __init__(self, waypoint_dir: str, movement_speed: int, rotation_speed: int):
        """Initialize controller with movement speeds and recording setup."""
        self.movement_speed = movement_speed  # cm/s for linear movements
        self.rotation_speed = rotation_speed  # degrees/s for rotations
        
        # Movement tracking state
        self.current_movement = None  # Active movement being recorded
        self.waypoints = []  # Complete waypoint sequence
        self.current_waypoint_movements = []  # Movements since last waypoint marker
        self.waypoint_counter = 0  # Sequential waypoint numbering
        
        # Control flags
        self.add_movement = False  # Whether to record current movement for waypoint data
        
        # Video streaming components (enabled for Windows)
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
                logger.log_debug('RealTimeDroneControllerWindows', f'Raw attitude response: {attitude_str}')
                
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
                                logger.log_warning('RealTimeDroneControllerWindows', f'Failed to parse yaw from "{part}": {e}')
                                continue
            except Exception as e:
                logger.log_warning('RealTimeDroneControllerWindows', f'Attitude query failed: {e}')
                state['yaw'] = 0

            # Get height in centimeters (converted from decimeters)
            try:
                height_str = drone_instance.send_command_with_return("height?", timeout=3)
                height_dm = int(height_str.replace('dm', ''))  # Remove 'dm' suffix
                state['height'] = height_dm * 10  # Convert dm to cm
            except Exception as e:
                logger.log_warning('RealTimeDroneControllerWindows', f'Height query failed: {e}')
                state['height'] = 0
            
            # Get battery percentage
            try:
                battery_str = drone_instance.send_command_with_return("battery?", timeout=3)
                state['battery'] = int(battery_str)
            except Exception as e:
                logger.log_warning('RealTimeDroneControllerWindows', f'Battery query failed: {e}')
                state['battery'] = 0
            

            return state
        except Exception as e:
            logger.log_error('RealTimeDroneControllerWindows', f'Error getting drone state: {e}')
            return {'height': 0, 'yaw': 0, 'battery': 0}  # Return safe defaults on error

    def start_movement(self, direction, movement_type="move", drone_instance=None):
        """Begin drone movement in specified direction and record movement data."""
        logger.log_debug('RealTimeDroneControllerWindows', f'Starting movement: {movement_type} {direction}')
        
        if self.current_movement is not None:
            logger.log_warning('RealTimeDroneControllerWindows', "Already moving, ignoring new movement")
            return  # Prevent overlapping movements that could cause conflicts
        
        # Get initial drone state for movement recording
        try: 
            drone_state = self.get_drone_state(drone_instance)
            start_yaw = drone_state.get('yaw', 0)  # Extract yaw for direction tracking
        except Exception as e:
            logger.log_error('RealTimeDroneControllerWindows', f'Error getting drone state: {e}')
            start_yaw = 0

        # Create movement record with timing and orientation
        self.current_movement = {
            'type': movement_type,
            'direction': direction,
            'start_time': time.time(),
            'start_yaw': start_yaw,
        }
        
        logger.log_debug('RealTimeDroneControllerWindows', f'Created movement record: {self.current_movement}')
        
        # Send appropriate RC control commands to drone
        try:
            if movement_type == "move":
                self.add_movement = True  # Linear movements are recorded for waypoint navigation

                logger.log_debug('RealTimeDroneControllerWindows', f'Sending RC control for {direction}')
                if direction == "forward":
                    drone_instance.send_rc_control(0, self.movement_speed, 0, 0)
                elif direction == "backward":
                    drone_instance.send_rc_control(0, -self.movement_speed, 0, 0)
                elif direction == "left":
                    drone_instance.send_rc_control(-self.movement_speed, 0, 0, 0)
                elif direction == "right":
                    drone_instance.send_rc_control(self.movement_speed, 0, 0, 0)
                logger.log_debug('RealTimeDroneControllerWindows', f'RC control sent for {direction}')

            elif movement_type == "lift":
                self.add_movement = True  # Vertical movements are recorded for waypoint navigation

                logger.log_debug('RealTimeDroneControllerWindows', f'Sending RC control for lift {direction}')
                if direction == "up":
                    drone_instance.send_rc_control(0, 0, self.movement_speed, 0)
                elif direction == "down":
                    drone_instance.send_rc_control(0, 0, -self.movement_speed, 0)
                logger.log_debug('RealTimeDroneControllerWindows', f'RC control sent for lift {direction}')
                    
            elif movement_type == "rotate":
                self.add_movement = False  # Rotations change orientation but don't create waypoint movements

                logger.log_debug('RealTimeDroneControllerWindows', f'Sending RC control for rotate {direction}')
                if direction == "anticlockwise":
                    drone_instance.send_rc_control(0, 0, 0, -self.rotation_speed)
                elif direction == "clockwise":
                    drone_instance.send_rc_control(0, 0, 0, self.rotation_speed)
                logger.log_debug('RealTimeDroneControllerWindows', f'RC control sent for rotate {direction}')
                    
        except Exception as e:
            logger.log_error('RealTimeDroneControllerWindows', f'Error starting movement: {e}')
            import traceback
            traceback.print_exc()
            self.current_movement = None
    
    def stop_movement(self, drone_instance=None):
        """Stop current drone movement and record distance traveled."""
        if self.current_movement is None:
            return  # No movement in progress
        
        # Handle rotation movements separately (no distance recording)
        if not self.add_movement:
            logger.log_debug('RealTimeDroneControllerWindows', 'Stopping rotation movement...')
            try:
                drone_instance.send_rc_control(0, 0, 0, 0)  # Stop all RC control
            except Exception as e:
                logger.log_error('RealTimeDroneControllerWindows', f'Error stopping rotation movement: {e}')
                
            self.current_movement = None  # Clear movement state
            return
        
        # Stop linear/vertical movements and record telemetry
        try:
            drone_instance.send_rc_control(0, 0, 0, 0)  # Stop all movement
        except Exception as e:
            logger.log_error('RealTimeDroneControllerWindows', f'Error stopping movement: {e}')
        
        # Calculate movement duration with halt delay compensation
        end_time = time.time()
        duration = end_time - self.current_movement['start_time'] + 0.5  # Add buffer for deceleration timing
        
        # Calculate distance moved based on speed and duration
        distance = self.movement_speed * duration  # Distance calculated in cm
        
        # Create detailed movement record with unique ID and timestamp
        movement_event = {
            'id': str(uuid.uuid4()),  # Unique identifier for this movement
            'type': self.current_movement['type'],
            'direction': self.current_movement['direction'],
            'distance': round(distance, 2),  # Round to 2 decimal places for precision
            'start_yaw': self.current_movement['start_yaw'],  # Drone orientation at movement start
            'timestamp': datetime.now().isoformat()  # ISO format timestamp for consistency
        }
        
        # Add movement to current waypoint's movement cluster
        self.current_waypoint_movements.append(movement_event)

        logger.log_info('RealTimeDroneControllerWindows', f"Recorded {movement_event['type']} {movement_event['direction']} at {movement_event['start_yaw']} degree(s): "
              f"{movement_event['distance']:.1f}cm")
        
        self.current_movement = None  # Clear movement state
    
    def mark_waypoint(self, name=None, auto_generated=False):
        """Mark current position as waypoint and save accumulated movements."""
        if not auto_generated and not name:
            name = input("Enter waypoint name: ").strip()
            if not name:
                name = f"Waypoint_{self.waypoint_counter + 1}"  # Auto-generate fallback name
        
        self.waypoint_counter += 1  # Increment waypoint counter
        waypoint_id = f"WP_{self.waypoint_counter:03d}"  # Format as WP_001, WP_002, etc.
        
        # Create waypoint record with movement history
        waypoint = {
            'id': waypoint_id,
            'name': name or f"Waypoint_{self.waypoint_counter}",
            'movements_to_here': self.current_waypoint_movements.copy()  # Copy movement list
        }
        
        self.waypoints.append(waypoint)  # Add to waypoint collection

        logger.log_info('RealTimeDroneControllerWindows', f"Waypoint marked: {waypoint['name']} (ID: {waypoint_id})")
        logger.log_info('RealTimeDroneControllerWindows', f"Movements recorded: {len(self.current_waypoint_movements)} events")

        # Reset movement list for next waypoint cluster
        self.current_waypoint_movements = []
    
    def save_to_json(self) -> list:
        """Convert waypoint data to JSON format with calculated yaws and movements."""
        processed_waypoints = []
        for waypoint in self.waypoints: 
            processed_movements = []

            # Process only 'move' and 'lift' movement types for waypoint calculation
            for movement in waypoint['movements_to_here']:
                if movement['type'] == 'move':
                    yaw = movement['start_yaw']  # Start with drone's initial orientation
                    
                    # Calculate yaw relative to drone's initial yaw based on drone's movement direction
                    if movement['direction'] == 'forward':
                        yaw += 0  # No change to yaw
                    elif movement['direction'] == 'backward':
                        yaw += 180  # Adjust yaw to opposite direction
                    elif movement['direction'] == 'left':
                        yaw -= 90  # Adjust yaw 90 degrees counter-clockwise
                    elif movement['direction'] == 'right':
                        yaw += 90  # Adjust yaw 90 degrees clockwise

                    # Normalize yaw to standard -180 to 180 degree range
                    if yaw > 180: 
                        yaw -= 360
                    elif yaw < -180:
                        yaw += 360
                    
                    # Create processed movement record with calculated yaw
                    processed_movement = {
                        'id': movement['id'],
                        'type': movement['type'],
                        'yaw': yaw,  # Yaw relative to drone's starting orientation
                        'distance': movement['distance'],
                        'timestamp': movement['timestamp']
                    }

                    processed_movements.append(processed_movement)
                
                else:  # Handle 'lift' movements (vertical)
                    # For vertical movements, preserve direction and distance (no yaw processing)
                    processed_movement = {
                        'id': movement['id'],
                        'type': movement['type'],
                        'direction': movement['direction'],  # 'up' or 'down'
                        'distance': movement['distance'],
                        'timestamp': movement['timestamp']
                    }
                    processed_movements.append(processed_movement)

            # Create processed waypoint with movement data
            processed_waypoint = {
                'id': waypoint['id'],
                'name': waypoint['name'],
                'movements_to_here': processed_movements
            }

            processed_waypoints.append(processed_waypoint)

        # Create final data structure with session metadata
        data = {
            'session_info': {
                'total_waypoints': len(self.waypoints),
            },
            'waypoints': processed_waypoints
        }
        
        # Create summary for return value
        summary = [{'id': wp['id'], 'name': wp['name']} for wp in processed_waypoints]

        # Save processed data to JSON file
        try:
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.log_success('RealTimeDroneControllerWindows', f'Data saved to {self.data_file}')
        except Exception as e:
            logger.log_error('RealTimeDroneControllerWindows', f"Error saving data: {e}")
            summary = []  # Return empty list on error
        finally: 
            return summary  # Return waypoint summary list
    
    def start_video_stream(self, drone_instance=None):
        """Initialize and start video streaming from drone camera."""
        try:
            logger.log_info('RealTimeDroneControllerWindows', 'Starting video stream...')
            
            # Enable video streaming on drone
            drone_instance.streamon()
            time.sleep(3)  # Allow stream initialization time
            
            # Get frame reader using djitellopy's built-in function
            self.frame_read = drone_instance.get_frame_read()
            
            # Wait for first valid frame to confirm stream is working
            retry_count = 0
            max_retries = 10
            while retry_count < max_retries:
                try:
                    test_frame = self.frame_read.frame
                    if test_frame is not None and test_frame.size > 0:
                        break  # Stream is working
                except:
                    pass  # Ignore frame errors during initialization
                retry_count += 1
                time.sleep(0.5)
                logger.log_debug('RealTimeDroneControllerWindows', f'Waiting for video stream... ({retry_count}/{max_retries})')
            
            # Check if stream initialization failed
            if retry_count >= max_retries:
                raise Exception("Video stream failed to initialize - no frames received")
            
            # Start background video display thread
            self.video_running = True
            self.video_thread = threading.Thread(target=self._video_display_loop, daemon=True)
            self.video_thread.start()
            
            logger.log_success('RealTimeDroneControllerWindows', 'Video stream started successfully.')
            logger.log_info('RealTimeDroneControllerWindows', 'Video stream window opened - you can see what the drone sees!')
            logger.log_info('RealTimeDroneControllerWindows', "Keep the video window visible to see the drone's perspective during mapping.")

        except Exception as e:
            logger.log_error('RealTimeDroneControllerWindows', 'Failed to initialize video stream, continuing mapping without video feed.')
            self.stop_video_stream(drone_instance=drone_instance)  # Clean up on failure

    def stop_video_stream(self, drone_instance=None):
        """Stop video streaming and clean up resources."""
        try:
            logger.log_info('RealTimeDroneControllerWindows', 'Stopping video stream...')
            
            # Stop video display thread
            self.video_running = False
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=2)  # Wait up to 2 seconds for thread to finish

            self.frame_read = None  # Clear frame reader reference
            
            # Close all OpenCV windows
            cv2.destroyAllWindows()
            
            # Disable drone video stream
            if drone_instance:
                drone_instance.streamoff()
            
            logger.log_success('RealTimeDroneControllerWindows', 'Video stream stopped.')
            
        except Exception as e:
            logger.log_error('RealTimeDroneControllerWindows', f'Error stopping video stream: {e}')
    
    def _video_display_loop(self):
        """Background thread for continuous video display with overlay information."""
        try:
            # Create OpenCV window for video display
            window_name = 'Drone Camera - Mapping Mode (Windows)'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            first_time = True  # Flag to handle first frame display
            logger.log_debug('RealTimeDroneControllerWindows', 'Video display thread started.')
            
            while self.video_running and self.frame_read:
                try:
                    # Get current frame from drone stream
                    frame = self.frame_read.frame
                    
                    if frame is not None and frame.size > 0:
                        # Resize frame if too large for display
                        height, width = frame.shape[:2]
                        if width > 960:  # Limit maximum display width
                            scale = 960 / width
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            frame = cv2.resize(frame, (new_width, new_height))
                        
                        # Create overlay for UI elements with transparency
                        overlay = frame.copy()
                        
                        # Draw header background (black with transparency)
                        cv2.rectangle(overlay, (0, 0), (width, 110), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                        
                        # Add title and control instructions
                        cv2.putText(frame, 'Drone Camera View - Mapping Mode (Windows)', 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, 'Use terminal for controls - Press Q in terminal to quit', 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Display frame in OpenCV window
                        cv2.imshow(window_name, frame)
                        
                        # Process window events without waiting for key presses
                        cv2.waitKey(1)

                        # Pin window to topmost on first display only, subsequent frame displays will use this same window
                        if first_time:
                            hwnd = win32gui.FindWindow(None, window_name)
                            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                            first_time = False # Subsequent frames will not re-pin the window and use the same window handle for display
                    
                    # Control frame rate to prevent excessive CPU usage
                    time.sleep(0.033)  # Approximately 30 FPS
                    
                except Exception as e:
                    logger.log_warning('RealTimeDroneControllerWindows', f'Frame display error: {e}')
                    time.sleep(0.1)  # Brief delay before retry
            
            logger.log_debug('RealTimeDroneControllerWindows', 'Video display thread ended.')
            
        finally:
            cv2.destroyAllWindows()  # Ensure OpenCV window cleanup
    
    def get_key(self):
        """Get single keypress with timeout - Windows implementation using msvcrt."""
        start_time = time.time()
        timeout = 0.2  # 200ms timeout for responsive input handling
        
        while time.time() - start_time < timeout:
            if msvcrt.kbhit():  # Check if key is available
                key = msvcrt.getch()
                
                # Handle special keys with escape sequences
                if key == b'\xe0' or key == b'\x00':  # Windows special key prefix for arrow keys
                    if msvcrt.kbhit():  # Check for following key code
                        key = msvcrt.getch()
                        # Map Windows arrow key codes to direction strings
                        arrow_map = {
                            b'H': 'up',    # Up arrow key
                            b'P': 'down',  # Down arrow key
                            b'M': 'right', # Right arrow key
                            b'K': 'left'   # Left arrow key
                        }
                        return arrow_map.get(key, 'unknown_key')
                    return 'incomplete'  # Incomplete escape sequence
                elif key == b'[':
                    # Ignore stray bracket character that may follow
                    if msvcrt.kbhit():
                        msvcrt.getch()
                    return 'ignored_key'
                else:
                    # Handle regular character keys
                    try:
                        return key.decode('utf-8').lower()  # Convert to lowercase string
                    except UnicodeDecodeError:
                        return 'unknown_key'
            time.sleep(0.01)  # Small delay to reduce CPU usage during polling
        return None  # No key pressed within timeout

    def handle_keypress(self, drone_instance=None):
        """Main keyboard input handler for Windows drone control using msvcrt."""
        
        try:
            activeMovementKey = None  # Track currently active movement key
            x_pressed = False  # Track waypoint marking state
            last_battery_check = 0  # Timestamp for battery monitoring
            
            print("🎮 Keyboard controls active!")  # User feedback
            
            while True:  # Main control loop
                # Periodic battery level monitoring
                current_time = time.time()
                if current_time - last_battery_check > 5:  # Check every 5 seconds
                    try:
                        battery_str = drone_instance.send_command_with_return("battery?", timeout=5)
                        battery = int(battery_str)
                        if battery < 20:
                            logger.log_warning('RealTimeDroneControllerWindows', f'Low battery: {battery}%')
                            if battery < 10:  # Critical battery level
                                logger.log_error('RealTimeDroneControllerWindows', 'CRITICAL: Battery too low, landing...')
                                break
                        last_battery_check = current_time
                    except Exception as e:
                        logger.log_error('RealTimeDroneControllerWindows', f'Error checking battery: {e}')

                # Get keyboard input with timeout
                key = self.get_key()
                
                if key:  # Key was pressed
                    logger.log_info('RealTimeDroneControllerWindows', f'Key: {key}')

                    if key == 'q':  # Quit mapping session
                        logger.log_info('RealTimeDroneControllerWindows', 'Finishing mapping session...')
                        break
                    elif key == 'x': 
                        if not x_pressed:
                            if self.current_movement:
                                self.stop_movement(drone_instance=drone_instance)
                                activeMovementKey = None

                            logger.log_info('RealTimeDroneControllerWindows', 'Marking waypoint...')
                            
                            self.mark_waypoint()
                            x_pressed = True
                        else:
                            logger.log_info('RealTimeDroneControllerWindows', 'Waypoint already marked')
                            continue
                    elif key in ['w', 'a', 's', 'd', 'up', 'down', 'left', 'right']:
                        x_pressed = False  # Reset waypoint flag on movement
                        if key != activeMovementKey:
                            # Stop current movement if any
                            if activeMovementKey:
                                logger.log_info('RealTimeDroneControllerWindows', f'Stopping movement: {activeMovementKey}')
                                self.stop_movement(drone_instance=drone_instance)

                            # Start new movement
                            activeMovementKey = key
                            logger.log_info('RealTimeDroneControllerWindows', f'Starting movement: {key}')

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
                            logger.log_info('RealTimeDroneControllerWindows', f'Continuing movement: {activeMovementKey}')
                            continue
                    else:
                        # Stop movement or remain still on other keys
                        logger.log_info('RealTimeDroneControllerWindows', f'Unrecognized key: {key}')
                        if self.current_movement:
                            logger.log_info('RealTimeDroneControllerWindows', 'Stopping current movement due to unrecognized key')
                            self.stop_movement(drone_instance=drone_instance)
                            activeMovementKey = None
                        continue
                else:
                    # No key pressed, stop any movement
                    if self.current_movement:
                        logger.log_info('RealTimeDroneControllerWindows', 'No key pressed, stopping current movement')
                        self.stop_movement(drone_instance=drone_instance)
                        activeMovementKey = None
                    continue
                
                time.sleep(0.02)  # Fast response loop - optimized for Windows timing
                
        except Exception as e:
            logger.log_error('RealTimeDroneControllerWindows', f'Error in keyboard handling: {e}')
        finally:
            logger.log_info('RealTimeDroneControllerWindows', 'Keyboard controls ended')


    def run(self, drone_instance=None) -> list:
        """Main control loop."""
        
        print("Starting keyboard control... Press Q to exit")
        
        # Mark the first waypoint automatically
        self.mark_waypoint("START", auto_generated=True)
        logger.log_info('RealTimeDroneControllerWindows', 'First waypoint marked: START')

        # Start video streaming for visual reference during mapping
        self.start_video_stream(drone_instance=drone_instance)

        try:
            self.handle_keypress(drone_instance=drone_instance)
        except KeyboardInterrupt:
            logger.log_info('RealTimeDroneControllerWindows', 'Keyboard interrupt received')
        except Exception as e:
            logger.log_error('RealTimeDroneControllerWindows', f'Error during drone control: {e}')
        finally:
            # Stop video streaming first
            self.stop_video_stream(drone_instance=drone_instance)
            
            # Ensure the last waypoint is marked if there are movements 
            try: 
                if self.current_movement:
                    self.stop_movement(drone_instance=drone_instance)
                
                # Mark final waypoint if there are pending movements
                if self.current_waypoint_movements:
                    self.mark_waypoint("END", auto_generated=True)

                # Save data to JSON file
                summary = self.save_to_json()

                print(f"\nSession complete! Data saved to: {self.data_file}")
            except Exception as e:
                logger.log_error('RealTimeDroneControllerWindows', f'Error finalizing session: {e}')
                summary = []

            return summary  # Return summary of waypoints

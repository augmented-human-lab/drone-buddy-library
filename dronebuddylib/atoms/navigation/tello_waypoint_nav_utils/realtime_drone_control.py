#!/usr/bin/env python3
import json
import os
import time
import uuid
import sys
import select
import termios
import tty
from datetime import datetime


class RealTimeDroneController:
    def __init__(self, waypoint_dir: str, movement_speed: int, rotation_speed: int):
        """Initialize the drone controller with recording capabilities."""
        self.movement_speed = movement_speed  # cm/s
        self.rotation_speed = rotation_speed  # degrees/s
        
        # Movement tracking
        self.current_movement = None
        self.waypoints = []
        self.current_waypoint_movements = []
        self.waypoint_counter = 0
        
        # Control flags
        self.add_movement = False
        
        # JSON file path for storing movement data
        filename = f"drone_movements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.data_file = os.path.join(waypoint_dir, filename)
    
    def get_drone_state(self, drone_instance=None):
        """Get current drone state including position and yaw."""
        try:
            state = {}

            # Get yaw (facing direction)
            try:
                attitude_str = drone_instance.send_command_with_return("attitude?", timeout=3)
                print(f"Raw attitude response: '{attitude_str}'")  # Debug line
                
                # Parse attitude string like "pitch:0;roll:0;yaw:45;"
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
                                print(f"⚠️  Failed to parse yaw from '{part}': {e}")
                                continue
            except Exception as e:
                print(f"⚠️  Attitude query failed: {e}")
                state['yaw'] = 0

            # Get height
            try:
                height_str = self.tello.send_command_with_return("height?", timeout=3)
                # Height returns like "10dm" (decimeters), convert to cm
                height_dm = int(height_str.replace('dm', ''))
                state['height'] = height_dm * 10  # Convert dm to cm
            except Exception as e:
                print(f"⚠️  Height query failed: {e}")
                state['height'] = 0
            
            # Get battery level
            try:
                battery_str = self.tello.send_command_with_return("battery?", timeout=3)
                state['battery'] = int(battery_str)
            except Exception as e:
                print(f"⚠️  Battery query failed: {e}")
                state['battery'] = 0
            

            return state
        except Exception as e:
            print(f"Error getting drone state: {e}")
            return {'height': 0, 'yaw': 0, 'battery': 0}

    def start_movement(self, direction, movement_type="move", drone_instance=None):
        """Start a movement in the specified direction."""
        print(f"🚀 start_movement called: {movement_type} {direction}")  # Debug
        
        if self.current_movement is not None:
            print("⚠️  Already moving, ignoring new movement")
            return  # Already moving
        
        try: 
            drone_state = self.get_drone_state(drone_instance)
            start_yaw = drone_state.get('yaw', 0)  # Default to 0 if not available
        except Exception as e:
            print(f"Error getting drone state: {e}")
            start_yaw = 0

        self.current_movement = {
            'type': movement_type,
            'direction': direction,
            'start_time': time.time(),
            'start_yaw': start_yaw,
        }
        
        print(f"📝 Created movement record: {self.current_movement}")  # Debug
        
        # Start the actual drone movement
        try:
            if movement_type == "move":
                self.add_movement = True

                print(f"📡 Sending RC control for {direction}")  # Debug
                if direction == "forward":
                    drone_instance.send_rc_control(0, self.movement_speed, 0, 0)
                elif direction == "backward":
                    drone_instance.send_rc_control(0, -self.movement_speed, 0, 0)
                elif direction == "left":
                    drone_instance.send_rc_control(-self.movement_speed, 0, 0, 0)
                elif direction == "right":
                    drone_instance.send_rc_control(self.movement_speed, 0, 0, 0)
                print(f"✅ RC control sent for {direction}")  # Debug
                    
            elif movement_type == "lift":
                self.add_movement = True

                print(f"📡 Sending RC control for lift {direction}")  # Debug
                if direction == "up":
                    drone_instance.send_rc_control(0, 0, self.movement_speed, 0)
                elif direction == "down":
                    drone_instance.send_rc_control(0, 0, -self.movement_speed, 0)
                print(f"✅ RC control sent for lift {direction}")  # Debug
                    
            elif movement_type == "rotate":
                self.add_movement = False

                print(f"📡 Sending RC control for rotate {direction}")  # Debug
                if direction == "anticlockwise":
                    drone_instance.send_rc_control(0, 0, 0, -self.rotation_speed)
                elif direction == "clockwise":
                    drone_instance.send_rc_control(0, 0, 0, self.rotation_speed)
                print(f"✅ RC control sent for rotate {direction}")  # Debug
                    
        except Exception as e:
            print(f"❌ Error starting movement: {e}")
            import traceback
            traceback.print_exc()
            self.current_movement = None
    
    def stop_movement(self, drone_instance=None):
        """Stop current movement and record the event."""
        if self.current_movement is None:
            return
        
        if not self.add_movement:
            print("Stopping rotation movement...")
            # Stop drone rotation
            try:
                drone_instance.send_rc_control(0, 0, 0, 0)
            except Exception as e:
                print(f"Error stopping rotation movement: {e}")
                
            self.current_movement = None
            return
        
        # Stop drone movement
        try:
            drone_instance.send_rc_control(0, 0, 0, 0)
        except Exception as e:
            print(f"Error stopping movement: {e}")
        
        # Calculate movement duration and distance
        end_time = time.time()
        duration = end_time - self.current_movement['start_time'] + 0.5 # Add a small buffer to account for halt delay
        
        # Calculate distance moved
        distance = self.movement_speed * duration  # cm
        
        # Create movement event record
        movement_event = {
            'id': str(uuid.uuid4()),
            'type': self.current_movement['type'],
            'direction': self.current_movement['direction'],
            'distance': round(distance, 2),
            'start_yaw': self.current_movement['start_yaw'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to current waypoint movements
        self.current_waypoint_movements.append(movement_event)
        
        print(f"Recorded {movement_event['type']} {movement_event['direction']} at {movement_event['start_yaw']} degree(s): "
              f"{movement_event['distance']:.1f}cm")
        
        self.current_movement = None
    
    def mark_waypoint(self, name=None, auto_generated=False):
        """Mark a waypoint and save current movement cluster."""
        if not auto_generated and not name:
            name = input("Enter waypoint name: ").strip()
            if not name:
                name = f"Waypoint_{self.waypoint_counter + 1}"
        
        self.waypoint_counter += 1
        waypoint_id = f"WP_{self.waypoint_counter:03d}"
        
        waypoint = {
            'id': waypoint_id,
            'name': name or f"Waypoint_{self.waypoint_counter}",
            'movements_to_here': self.current_waypoint_movements.copy()
        }
        
        self.waypoints.append(waypoint)
        
        print(f"Waypoint marked: {waypoint['name']} (ID: {waypoint_id})")
        print(f"Movements recorded: {len(self.current_waypoint_movements)} events")
        
        # Reset movements for next waypoint cluster
        self.current_waypoint_movements = []
    
    def save_to_json(self) -> list:
        """Save all waypoints and movements to JSON file."""
        processed_waypoints = []
        for waypoint in self.waypoints: 
            processed_movements = []

            # Movement type is either 'move', or 'lift' only
            for movement in waypoint['movements_to_here']:
                if movement['type'] == 'move':
                    yaw = movement['start_yaw']
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
                    # For 'lift' movements, we can just record the type distance and direction
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
            print(f"Data saved to {self.data_file}")
        except Exception as e:
            print(f"Error saving data: {e}")
            summary = []
        finally: 
            # Return a summary list of all waypoint ids and names 
            return summary
    
    def get_key(self):
        """Get a single key press without blocking."""
        if select.select([sys.stdin], [], [], 0.5) == ([sys.stdin], [], []):
            # Read a single character from stdin
            key = sys.stdin.read(1).lower()
            
            if key == '\x1b':

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
                # Ignore the alphebet key character that follows
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    sys.stdin.read(1)
                return 'ignored_key'  
            else: 
                return key  # Regular key press
        return None

    def handle_keypress(self, drone_instance=None):
        """Handle keyboard input for drone control using termios."""
        
        # Save original terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            # Set terminal to raw mode for immediate key detection
            tty.setraw(sys.stdin)
            
            activeMovementKey = None
            x_pressed = False
            last_battery_check = 0
            
            print("🎮 Keyboard controls active!")
            
            while True:
                # Battery check every 5 seconds
                current_time = time.time()
                if current_time - last_battery_check > 5:
                    try:
                        battery_str = drone_instance.send_command_with_return("battery?", timeout=5)
                        battery = int(battery_str)
                        if battery < 20:
                            print(f"\r⚠️  Low battery ({battery}%)               ")
                            if battery < 10:
                                print("\r❗ CRITICAL: Battery too low, landing...")
                                break
                        last_battery_check = current_time
                    except Exception as e:
                        print(f"\rError checking battery: {e}")
                
                # Get key input
                key = self.get_key()
                
                if key:
                    print(f"\r🎮 Key: '{key}'                    ")
                    
                    if key == 'q':  
                        print("\r--- Finishing mapping session ---")
                        break
                    elif key == 'x': 
                        if not x_pressed:
                            if self.current_movement:
                                self.stop_movement(drone_instance=drone_instance)
                                activeMovementKey = None

                            print("\r--- Marking Waypoint ---")
                            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

                            self.mark_waypoint()

                            old_settings = termios.tcgetattr(sys.stdin)
                            tty.setraw(sys.stdin)
                            x_pressed = True
                        else:
                            print("\r--- Waypoint already marked ---")
                            continue
                    elif key in ['w', 'a', 's', 'd', 'up', 'down', 'left', 'right']:
                        x_pressed = False  # Reset x_pressed flag
                        if key != activeMovementKey:
                            # Stop current movement if any
                            if activeMovementKey:
                                print(f"\r🛑 Stopping movement: {activeMovementKey}        ")
                                self.stop_movement(drone_instance=drone_instance)

                            # Start new movement
                            activeMovementKey = key
                            print(f"\r🚀 Starting movement: {key}        ")
                            
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
                            print(f"\r🚀 Continuing movement: {activeMovementKey}    ")
                            continue
                    else:
                        # Stop movement or remain still on other keys
                        print(f"\r🎮 Unrecognized key: '{key}'                  ")
                        if self.current_movement:
                            print("\r🛑 Stopping current movement due to unrecognized key")
                            self.stop_movement(drone_instance=drone_instance)
                            activeMovementKey = None
                        continue
                else:
                    # No key pressed, stop any movement
                    if self.current_movement:
                        print("\r🛑 No key pressed, stopping current movement    ")
                        self.stop_movement(drone_instance=drone_instance)
                        activeMovementKey = None
                    continue
                
                time.sleep(0.05)  # Fast responsive loop
                
        except Exception as e:
            print(f"\rError in keyboard handling: {e}")
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            print("\r🎮 Keyboard controls ended            ")
    
    
    def run(self, drone_instance=None) -> list:
        """Main control loop."""
        
        print("Starting keyboard control... Press ESC to exit")
        
        # Mark the first waypoint automatically
        self.mark_waypoint("START", auto_generated=True)
        print("First waypoint marked: START")

        try:
            self.handle_keypress(drone_instance=drone_instance)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
        except Exception as e:
            print(f"\n❌ Error during drone control: {e}")
        finally:
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
                print(f"Error finalizing session: {e}")
                summary = []

            return summary  # Return summary of waypoints

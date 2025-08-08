from djitellopy import Tello

from dronebuddylib.utils.logger import Logger
import time
import cv2
import os
from typing import Optional
from datetime import datetime
from PIL import Image

logger = Logger()

class TelloNavExtra:
    def __init__(self, tello: Tello = None, image_dir: str = None):
        self.tello = tello
        self.image_dir = image_dir

    def forward(self, distance: float) -> bool:
        """
        Moves the Tello drone forward by a specified distance.

        Args:
            distance (float): The distance to move forward in centimeters.
        """
        if self.tello is not None: 
            return self._move_forward(distance)
        else:
            self.tello = Tello()
            if self._connect_drone():
                if self._takeoff():
                    if self._move_forward(distance): 
                        logger.log_success('TelloNavExtra', f'Moved forward {distance} cm successfully.')
                        if self._land():
                            logger.log_success('TelloNavExtra', 'Drone landed successfully after moving forward.')
                            return True
                        else: 
                            logger.log_error('TelloNavExtra', 'Landing failed after moving forward.')
                            return False
                    else:
                        logger.log_error('TelloNavExtra', 'Moving forward failed.')
                        return False
                else:
                    logger.log_error('TelloNavExtra', 'Takeoff failed, cannot move forward.')
                    return False
            else:
                logger.log_error('TelloNavExtra', 'Failed to connect to Tello drone.')
                return False

    def scan(self, current_waypoint_file: str,current_waypoint: str) -> list:
        """
        Scans the surrounding of the drone while doing a 360 degree rotation.
        Captures images at 15-degree intervals (24 total images) and saves them with yaw metadata.

        Args:
            current_waypoint (str): The current waypoint of the drone.
        
        Returns:
            list: List of dictionaries containing image info:
                  [{'image_path': str, 'filename': str, 'waypoint_file': str, 'waypoint': str, 'rotation_from_start': str, 'image_number': int, 'timestamp': str, 'format': str='JPEG'}, ...]
        """
        try:
            logger.log_info('TelloNavExtra', f'Starting 360-degree scan at waypoint: {current_waypoint}')

            # Scan configuration parameters
            ROTATION_INTERVAL = 15  # degrees (24 images total for 360°)
            TOTAL_ROTATION = 360
            STABILIZATION_TIME = 0.5  # seconds to wait after rotation before capture
            
            # Setup image storage
            base_dir = self._setup_image_storage_directory(current_waypoint_file, current_waypoint)

            # Initialize scan results list and initial drone yaw
            scan_results = []
            initial_yaw = self.get_yaw()
            
            # Start video stream
            logger.log_info('TelloNavExtra', 'Starting video stream for image capture...')
            self.tello.streamon()
            
            # Get frame reader
            frame_read = self.tello.get_frame_read()
            if frame_read is None:
                logger.log_error('TelloNavExtra', 'Failed to get frame reader')
                return []
            
            # Execute 360-degree rotation scan
            images_captured = 0
            current_rotation = 0
            
            while current_rotation < TOTAL_ROTATION:
                # Wait for drone stabilization at current angle
                logger.log_debug('TelloNavExtra', f'Stabilizing at {current_rotation}° clockwise rotation relative to drones initial position at current waypoint {current_waypoint}')
                time.sleep(STABILIZATION_TIME)

                # check battery level before capturing image
                try: 
                    battery_str = self.tello.send_command_with_return("battery?", timeout=3)
                    logger.log_debug('TelloNavExtra', 'checking battery status')
                    battery = int(battery_str)
                    if battery < 20:
                        logger.log_warning('TelloNavExtra', f'Low battery detected: {battery}%')

                        # Exit scan if battery is critically low
                        if battery < 10:
                            logger.log_error('TelloNavExtra', f'CRITICAL: Battery too low ({battery}%), stopping scan.')
                            try:
                                self.tello.streamoff()
                            except:
                                logger.log_error('TelloNavExtra', 'Failed to stop video stream after scan failure')
                            return scan_results  
                except Exception as e:
                    logger.log_error('TelloNavExtra', f'Failed to check battery status: {e}')
                    pass  # Continue scan even if battery check fails
                    
                try:
                    # Capture current frame
                    frame = frame_read.frame
                    if frame is not None and frame.size > 0:
                        images_captured += 1

                        # Save image with metadata
                        image_info = self._save_scan_image(
                            frame, 
                            current_waypoint_file, 
                            current_waypoint, 
                            current_rotation,
                            images_captured,
                            base_dir
                        )
                        
                        # If image was saved successfully, add to results and log success, else log error
                        if image_info:
                            scan_results.append(image_info)
                            logger.log_success('TelloNavExtra', f'Captured image {images_captured}/24 at clockwise rotation {current_rotation}° relative to drones initial position at current waypoint {current_waypoint}')
                        else: 
                            logger.log_error('TelloNavExtra', f'Failed to save image at rotation {current_rotation}°')
                    else:
                        # Log warning if frame is not available and continue scan
                        logger.log_warning('TelloNavExtra', f'No frame available at rotation {current_rotation}°')

                except Exception as e:
                    # Log error if image capture fails and continue scan
                    logger.log_error('TelloNavExtra', f'Error during scan at {current_rotation}°: {e}')
                    continue
                finally: 
                    try: 
                        # Advance to next rotation position
                        if current_rotation + ROTATION_INTERVAL <= TOTAL_ROTATION:
                            logger.log_debug('TelloNavExtra', f'Rotating {ROTATION_INTERVAL}° clockwise...')
                            self.tello.rotate_clockwise(ROTATION_INTERVAL + 1) # Compensation factor - drone always rotate < 15 degrees 
                    except Exception as e:
                        logger.log_error('TelloNavExtra', f'Failed to rotate clockwise: {e}')
                        pass # Continue to next scan
                    
                    # Increment rotation accumulator regardless of success
                    current_rotation += ROTATION_INTERVAL
            
            # Stop video stream
            try: 
                self.tello.streamoff()
            except: 
                logger.log_error('TelloNavExtra', 'Failed to stop video stream')
            
            
            logger.log_success('TelloNavExtra', f'Scan completed! Captured {images_captured} images at waypoint: {current_waypoint} of file: {current_waypoint_file}')
            logger.log_info('TelloNavExtra', f'Images saved in: {base_dir}')

            # Check if we need to return to initial drone yaw after scan completion
            current_yaw = self.get_yaw()
            if initial_yaw is not None and current_yaw is not None and initial_yaw != current_yaw: 
                # Attempt to return to initial drone yaw
                success = self.return_initial_yaw(current_yaw, initial_yaw)

                if success:
                    logger.log_success('TelloNavExtra', f'Returned to initial yaw {initial_yaw}° successfully after scan.')
                else:
                    logger.log_error('TelloNavExtra', f'Failed to return to initial yaw {initial_yaw}° after scan. Continuing with current yaw {current_yaw}°.')
            
            return scan_results # Return list of captured images with metadata
            
        except Exception as e:
            logger.log_error('TelloNavExtra', f'Scan failed: {e}')
            # Ensure video stream is stopped on error and return any accumulated scan results
            try:
                self.tello.streamoff()
            except:
                logger.log_error('TelloNavExtra', 'Failed to stop video stream after scan failure')
            return scan_results
    
    def _setup_image_storage_directory(self, waypoint_file: str, waypoint: str) -> str:
        """
        Create directory structure for storing scan images.
        
        Structure: ~/dronebuddylib/scans/{waypoint_file_name}/{waypoint}_{timestamp}/
        
        Args:
            waypoint_file (str): Name of the waypoint file
            waypoint (str): Current waypoint name
            
        Returns:
            str: Full path to the created directory
        """
        if self.image_dir is not None and os.path.exists(self.image_dir):
            # Use provided custom directory if valid
            base_scans_dir = self.image_dir
        else:
            # Fall back to default home directory structure
            home_dir = os.path.expanduser("~")
            base_scans_dir = os.path.join(home_dir, "dronebuddylib", "scans")
            self.image_dir = base_scans_dir  # Update instance variable
        
        # Create timestamped directory for this scan
        waypoint_file_name = os.path.splitext(waypoint_file)[0]  # Remove file extension
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scan_dir = os.path.join(base_scans_dir, f"{waypoint_file_name}", f"{waypoint}_{timestamp}")
        
        # Ensure directory exists
        os.makedirs(scan_dir, exist_ok=True)

        logger.log_info('TelloNavExtra', f'Created scan directory: {scan_dir}')
        return scan_dir
    
    def _save_scan_image(self, frame, waypoint_file: str, waypoint: str, rotation: int, image_number: int, base_dir: str) -> dict:
        """
        Save captured frame as JPEG with metadata.
        
        Args:
            frame: OpenCV frame from drone camera
            waypoint (str): Current waypoint name
            yaw (float): Drone's yaw angle in degrees
            rotation (int): Rotation angle from start position
            image_number (int): Sequential image number
            base_dir (str): Base directory for saving
            
        Returns:
            dict: Image information with metadata
        """
        try:
            # Build unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            filename = f"{waypoint}_scan_{image_number:02d}_rotation_{rotation}_{timestamp}.jpg"
            image_path = os.path.join(base_dir, filename)
            
            # Convert OpenCV BGR format to PIL RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Save with high quality JPEG compression
            pil_image.save(image_path, 'JPEG', quality=95, optimize=True)
            
            # Build image metadata record
            image_info = {
                'image_path': image_path,
                'filename': filename,
                'waypoint_file': waypoint_file,
                'waypoint': waypoint, 
                'rotation_from_start': rotation,
                'image_number': image_number,
                'timestamp': timestamp,
                'format': 'JPEG'
            }

            logger.log_debug('TelloNavExtra', f'Saved image: {filename}')
            return image_info
            
        except Exception as e:
            logger.log_error('TelloNavExtra', f'Failed to save image: {e}')
            return None
    
    def get_yaw(self) -> Optional[int]:
        """
        Get current drone yaw angle from attitude telemetry.
        
        Returns: 
            int: Yaw angle in degrees, or None if unable to retrieve.
        """
        try:
            # Query Tello for attitude data
            attitude_str = self.tello.send_command_with_return("attitude?", timeout=3)
            logger.log_debug('TelloNavExtra', f'Raw attitude response: {attitude_str}')
            
            # Parse attitude string format: "pitch:0;roll:0;yaw:45; to extract yaw"
            yaw = None  # Default fallback value
            if attitude_str and ':' in attitude_str:
                attitude_parts = attitude_str.split(';')
                for part in attitude_parts:
                    if part.strip() and 'yaw:' in part:
                        try:
                            yaw_value = part.split(':')[1].strip()
                            if yaw_value:
                                yaw = int(yaw_value)
                        except (ValueError, IndexError) as e:
                            logger.log_warning('TelloNavExtra', f'Failed to parse yaw from "{part}": {e}')
                            continue
            return yaw # Return extracted yaw angle or None if parsing failed
        except Exception as e:
            logger.log_warning('TelloNavExtra', f'Attitude query failed: {e}')
            return None  # Return error yaw indication on communication error
    
    def return_initial_yaw(self, current_yaw: int, initial_yaw: int) -> bool: 
        """ 
        Adjust the drone's yaw to return to the initial yaw angle.
        This method calculates the shortest rotation path to return to the initial yaw angle.

        Args:
            current_yaw (int): The current yaw angle of the drone.
            initial_yaw (int): The initial yaw angle to return to.

        Returns:
            bool: True if the yaw adjustment was successful, False otherwise.
        """
        logger.log_info('TelloNavExtra', f'Adjusting yaw from {current_yaw} back to initial yaw {initial_yaw}')

        # Calculate the absolute difference in yaw
        turn_degree = abs(initial_yaw - current_yaw)

        try: 
            # Calculate required yaw adjustment for shortest rotation path and execute it
            if current_yaw > initial_yaw:
                if turn_degree > 180 and turn_degree < 360:
                    self.tello.rotate_clockwise(360 - turn_degree)  # Shorter rotation path
                elif turn_degree <= 180 and turn_degree > 0: 
                    self.tello.rotate_counter_clockwise(turn_degree)
                else: 
                    logger.log_debug('TelloNavExtra', 'No yaw adjustment needed')
                self.tello.send_rc_control(0, 0, 0, 0)  # Stop rotation
            else: 
                if turn_degree > 180 and turn_degree < 360: 
                    self.tello.rotate_counter_clockwise(360 - turn_degree)  # Shorter rotation path
                elif turn_degree <= 180 and turn_degree > 0: 
                    self.tello.rotate_clockwise(turn_degree)
                else: 
                    logger.log_debug('TelloNavExtra', 'No yaw adjustment needed')
                self.tello.send_rc_control(0, 0, 0, 0)  # Stop rotation
            
            return True # Return True if yaw adjustment was successful
        except Exception as e:
            logger.log_error('TelloNavExtra', f'Failed to adjust yaw: {e}')
            return False # Return False if yaw adjustment failed
    
    def _move_forward(self, distance: float) -> bool:
        """
        Moves the Tello drone forward by a specified distance in a blocking manner.

        Args:
            distance (float): The distance to move forward in centimeters.
        """ 
        try: 
            self.tello.move_forward(distance)
            return True
        except Exception as e:
            print(f"Error occurred while moving forward: {e}")
            return False
    
    def _connect_drone(self):
        """
        Establish connection to DJI Tello drone with configuration and status validation.
        """
        try:
            logger.log_info('TelloNavExtra', 'Connecting to Tello drone...')
            self.tello.RESPONSE_TIMEOUT = 7
            self.tello.connect(wait_for_state=False)
            logger.log_success('TelloNavExtra', 'Drone connected successfully!')

            try:
                battery_response = self.tello.send_command_with_return("battery?", timeout=5)
                logger.log_info('TelloNavExtra', f'Battery: {battery_response}%')
            except Exception as e:
                logger.log_error('TelloNavExtra', f'Battery command failed: {e}')

            return True
        except Exception as e:
            logger.log_error('TelloNavExtra', f'Failed to connect to drone: {e}')
            return False
    
    def _takeoff(self):
        """
        Execute drone takeoff sequence with safety validation and stabilization.
       """
        try:
            logger.log_info('TelloNavExtra', 'Taking off...')
            self.tello.takeoff()
            self.is_flying = True
            time.sleep(1)  # Stabilization delay
            logger.log_success('TelloNavExtra', 'Drone is airborne!')
            return True
        
        except Exception as e:
            logger.log_error('TelloNavExtra', f'Takeoff failed: {e}')
            return False
    
    def _land(self):
        """
        Execute safe drone landing sequence with status management.
        """
        try:
            logger.log_info('TelloNavExtra', 'Landing drone...')
            self.tello.land()
            self.is_flying = False
            logger.log_success('TelloNavExtra', 'Drone landed successfully!')
        except Exception as e:
            logger.log_error('TelloNavExtra', f'Landing failed: {e}')
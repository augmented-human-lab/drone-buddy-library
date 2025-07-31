from djitellopy import Tello
from .tello_waypoint_nav_coordinator import TelloWaypointNavCoordinator

from dronebuddylib.utils.logger import Logger
import time
import cv2
import os
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS

logger = Logger()

class TelloNavExtra:
    def __init__(self, tello: Tello = None, image_dir: str = None):
        self.tello = tello
        self.image_dir = image_dir

    # def forward(self, distance: float) -> bool:
    #     """
    #     Moves the Tello drone forward by a specified distance.

    #     Args:
    #         distance (float): The distance to move forward in centimeters.
    #     """
    #     if self.tello is not None: 
    #         return self._move_forward(distance)
    #     else:
    #         self.tello = Tello()
    #         if self._connect_drone():
    #             if self._takeoff():
    #                 if self._move_forward(distance): 
    #                     logger.log_success('TelloNavDirect', f'Moved forward {distance} cm successfully.')
    #                     if self._land():
    #                         logger.log_success('TelloNavDirect', 'Drone landed successfully after moving forward.')
    #                         return True
    #                     else: 
    #                         logger.log_error('TelloNavDirect', 'Landing failed after moving forward.')
    #                         return False
    #                 else:
    #                     logger.log_error('TelloNavDirect', 'Moving forward failed.')
    #                     return False
    #             else:
    #                 logger.log_error('TelloNavDirect', 'Takeoff failed, cannot move forward.')
    #                 return False
    #         else:
    #             logger.log_error('TelloNavDirect', 'Failed to connect to Tello drone.')
    #             return False

    def scan(self, current_waypoint_file: str,current_waypoint: str) -> list:
        """
        Scans the surrounding of the drone while doing a 360 degree rotation.
        Captures images at 15-degree intervals (24 total images) and saves them with yaw metadata.

        Args:
            current_waypoint (str): The current waypoint of the drone.
        
        Returns:
            list: List of dictionaries containing image info:
                  [{'image_path': str, 'yaw': float, 'timestamp': str, 'waypoint': str}, ...]
        """
        try:
            logger.log_info('TelloNavDirect', f'Starting 360-degree scan at waypoint: {current_waypoint}')
            
            # Configuration
            ROTATION_INTERVAL = 15  # degrees (24 images total for 360°)
            TOTAL_ROTATION = 360
            STABILIZATION_TIME = 0.5  # seconds to wait after rotation before capture
            
            # Setup image storage
            scan_results = []
            base_dir = self._setup_image_storage_directory(current_waypoint_file, current_waypoint)
            
            # Start video stream
            logger.log_info('TelloNavDirect', 'Starting video stream for image capture...')
            self.tello.streamon()
            time.sleep(1)  # Allow stream to stabilize
            
            # Get frame reader
            frame_read = self.tello.get_frame_read()
            if frame_read is None:
                logger.log_error('TelloNavDirect', 'Failed to get frame reader')
                return []
            
            # Perform rotation scan
            images_captured = 0
            current_rotation = 0
            
            while current_rotation < TOTAL_ROTATION:
                # Stabilize before capture
                logger.log_debug('TelloNavDirect', f'Stabilizing at {current_rotation}° clockwise rotation relative to drones initial position at current waypoint {current_waypoint}')
                time.sleep(STABILIZATION_TIME)
                try:
                    # Capture frame
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
                        
                        if image_info:
                            scan_results.append(image_info)
                            logger.log_success('TelloNavDirect', f'Captured image {images_captured}/24 at clockwise rotation {current_rotation}° relative to drones initial position at current waypoint {current_waypoint}')
                        else: 
                            logger.log_error('TelloNavDirect', f'Failed to save image at rotation {current_rotation}°')
                    else:
                        logger.log_warning('TelloNavDirect', f'No frame available at rotation {current_rotation}°')
                    
                except Exception as e:
                    logger.log_error('TelloNavDirect', f'Error during scan at {current_rotation}°: {e}')
                    continue
                finally: 
                    # Move to next position
                    if current_rotation + ROTATION_INTERVAL <= TOTAL_ROTATION:
                        logger.log_debug('TelloNavDirect', f'Rotating {ROTATION_INTERVAL}° clockwise...')
                        self.tello.rotate_clockwise(ROTATION_INTERVAL)
                    
                    current_rotation += ROTATION_INTERVAL
            
            # Stop video stream
            try: 
                self.tello.streamoff()
            except: 
                logger.log_error('TelloNavDirect', 'Failed to stop video stream')
            
            
            logger.log_success('TelloNavDirect', f'Scan completed! Captured {images_captured} images at waypoint: {current_waypoint} of file: {current_waypoint_file}')
            logger.log_info('TelloNavDirect', f'Images saved in: {base_dir}')
            
            return scan_results
            
        except Exception as e:
            logger.log_error('TelloNavDirect', f'Scan failed: {e}')
            # Ensure video stream is stopped on error
            try:
                self.tello.streamoff()
            except:
                logger.log_error('TelloNavDirect', 'Failed to stop video stream after scan failure')
            return []
    
    def _setup_image_storage_directory(self, waypoint_file: str, waypoint: str) -> str:
        """
        Create directory structure for storing scan images.
        
        Structure: ~/dronebuddylib/scans/{waypoint}_{timestamp}/
        
        Args:
            waypoint (str): Current waypoint name
            
        Returns:
            str: Full path to the created directory
        """
        if self.image_dir is not None and os.path.exists(self.image_dir):
            # Use provided image directory if it exists
            base_scans_dir = self.image_dir
        else:
            # Create base directory in user home
            home_dir = os.path.expanduser("~")
            base_scans_dir = os.path.join(home_dir, "dronebuddylib", "scans")
            self.image_dir = base_scans_dir  # Update instance variable
        
        # Create timestamped directory for this scan
        waypoint_file_name = os.path.splitext(waypoint_file)[0]  # Remove file extension
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scan_dir = os.path.join(base_scans_dir, f"{waypoint_file_name}", f"{waypoint}_{timestamp}")
        
        # Ensure directory exists
        os.makedirs(scan_dir, exist_ok=True)
        
        logger.log_info('TelloNavDirect', f'Created scan directory: {scan_dir}')
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
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            filename = f"{waypoint}_scan_{image_number:02d}_rotation_{rotation}_{timestamp}.jpg"
            image_path = os.path.join(base_dir, filename)
            
            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Save as JPEG with high quality
            pil_image.save(image_path, 'JPEG', quality=95, optimize=True)
            
            # Create metadata dictionary
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
            
            logger.log_debug('TelloNavDirect', f'Saved image: {filename}')
            return image_info
            
        except Exception as e:
            logger.log_error('TelloNavDirect', f'Failed to save image: {e}')
            return None
    
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
            logger.log_info('TelloWaypointNavCoordinator', 'Connecting to Tello drone...')
            self.tello.RESPONSE_TIMEOUT = 7
            self.tello.connect(wait_for_state=False)
            logger.log_success('TelloWaypointNavCoordinator', 'Drone connected successfully!')

            try:
                battery_response = self.tello.send_command_with_return("battery?", timeout=5)
                logger.log_info('TelloWaypointNavCoordinator', f'Battery: {battery_response}%')
            except Exception as e:
                logger.log_error('TelloWaypointNavCoordinator', f'Battery command failed: {e}')
            
            return True
        except Exception as e:
            logger.log_error('TelloWaypointNavCoordinator', f'Failed to connect to drone: {e}')
            return False
    
    def _takeoff(self):
        """
        Execute drone takeoff sequence with safety validation and stabilization.
       """
        try:
            logger.log_info('TelloWaypointNavCoordinator', 'Taking off...')
            self.tello.takeoff()
            self.is_flying = True
            time.sleep(1)  # Wait for stabilization
            logger.log_success('TelloWaypointNavCoordinator', 'Drone is airborne!')
            return True
        
        except Exception as e:
            logger.log_error('TelloWaypointNavCoordinator', f'Takeoff failed: {e}')
            return False
    
    def _land(self):
        """
        Execute safe drone landing sequence with status management.
        """
        try:
            logger.log_info('TelloWaypointNavCoordinator', 'Landing drone...')
            self.tello.land()
            self.is_flying = False
            logger.log_success('TelloWaypointNavCoordinator', 'Drone landed successfully!')
        except Exception as e:
            logger.log_error('TelloWaypointNavCoordinator', f'Landing failed: {e}')
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import NavigationAlgorithm
from dronebuddylib.utils.logger import Logger
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dronebuddylib.atoms.navigation.tello_waypoint_nav_utils.tello_nav_extra import ScanResult

logger = Logger()

class NavigationEngine:
    
    def __init__(self, algorithm: NavigationAlgorithm, config: EngineConfigurations):
        """
        Initializes the navigation engine with the specified algorithm and configuration.

        Args:
            algorithm (NavigationAlgorithm): The navigation algorithm to be used.
            config (EngineConfigurations): The configuration for the navigation engine.
        """
        if algorithm == NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT or algorithm == NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT.name:
            logger.log_info(self.get_class_name(), 'Preparing to initialize Tello Waypoint navigation engine.')
            from dronebuddylib.atoms.navigation.tello_navigation_impl import NavigationWaypointImpl
            self.navigation_engine = NavigationWaypointImpl(config)
            logger.log_debug(self.get_class_name(), 'Tello Waypoint navigation engine initialized successfully.')
        else:
            logger.log_error(self.get_class_name(), f'Unsupported navigation algorithm: {algorithm}')
            raise ValueError(f"Unsupported navigation algorithm: {algorithm}")

    def get_class_name(self) -> str:
        """
        Returns the class name.

        Returns:
            str: The class name.
        """
        return 'NAVIGATION_ENGINE'

    def map_location(self) -> list:
        """
        Allows user to map the current location and returns a list of waypoints.

        Returns:
            list: A list of waypoints representing the mapped location.
        """
        logger.log_debug(self.get_class_name(), 'Starting map location operation.')
        result = self.navigation_engine.map_location()
        logger.log_debug(self.get_class_name(), f'Map location operation completed. Mapped {len(result)} waypoints.')
        return result

    def navigate(self) -> list:
        """
        Provides navigation interface to the user to navigate between known waypoints
        
        Returns: 
         list: A list of navigated waypoints.
        """
        logger.log_info(self.get_class_name(), 'Starting navigation operation.')
        result = self.navigation_engine.navigate()
        logger.log_debug(self.get_class_name(), f'Navigation operation completed with {len(result)} results.')
        return result

    def navigate_to_waypoint(self, destination_waypoint, instruction) -> list:
        """
        Navigates to a specific waypoint with strict NavigationInstruction enum enforcement.

        Args:
            destination_waypoint (str): The waypoint to navigate to.
            instruction (NavigationInstruction): Must be NavigationInstruction.CONTINUE or NavigationInstruction.HALT.

        Returns:
            list: Result of the navigation operation, first element is a boolean indicating if the drone has landed or not (True if landed, False if still flying), second element is the current waypoint of the drone. Can be used to determine if the drone has successfully navigated to the destination waypoint.

        Raises:
            TypeError: If instruction is not a NavigationInstruction enum.
        """
        
        logger.log_info(self.get_class_name(), f'Starting navigation to waypoint: {destination_waypoint}')
        logger.log_debug(self.get_class_name(), f'Navigation instruction: {instruction}')
        
        result = self.navigation_engine.navigate_to_waypoint(destination_waypoint, instruction)
        
        logger.log_debug(self.get_class_name(), f'Navigate to waypoint operation completed with drone at current waypoint: {result[0]}.')
        return result
    
    def navigate_to(self, waypoints, final_instruction): 
        """
        Navigates to a sequence of waypoints with strict NavigationInstruction enum enforcement.

        Args:
            waypoints (list): List of waypoints to navigate to.
            final_instruction (NavigationInstruction): Must be NavigationInstruction.CONTINUE or NavigationInstruction.HALT.

        Returns:
            list: Contains the list of waypoints the drone has navigated to. 
            
        Raises:
            TypeError: If final_instruction is not a NavigationInstruction enum.
        """
        
        logger.log_info(self.get_class_name(), f'Starting navigation to waypoints: {waypoints}')
        logger.log_debug(self.get_class_name(), f'Final navigation instruction: {final_instruction}')
        
        result = self.navigation_engine.navigate_to(waypoints, final_instruction)
        
        logger.log_debug(self.get_class_name(), f'Navigate to waypoints operation completed with drone at current waypoint: {result[0]}.')
        return result
    
    def scan_surrounding(self) -> list:
        """
        Run a basic 360 scan and return captured images.

        Returns:
            list: A list of images captured during the scan.
        """
        
        logger.log_info(self.get_class_name(), 'Starting surrounding scan operation.')

        result = self.navigation_engine.scan_surrounding()

        logger.log_debug(self.get_class_name(), f'Surrounding scan operation completed with {len(result)} images captured.')
        return result
    
    def scan_with_detection(
        self, 
        target_object: Optional[str] = None,
        yolo_model_path: Optional[str] = None,
        yolo_conf_threshold: float = 0.25,
        yolo_iou_threshold: float = 0.45
    ) -> 'ScanResult':
        """
        Run a 360 scan and apply standard YOLO detection on each frame.
        
        Args:
            target_object: Specific object to search for (e.g., "cup", "bottle").
                          If provided, the result will indicate if this object was found.
            yolo_model_path: Path to ONNX YOLO model file.
            yolo_conf_threshold: YOLO confidence threshold (default: 0.25)
            yolo_iou_threshold: YOLO IOU threshold for NMS (default: 0.45)
            
        Returns:
            ScanResult: Structured result containing:
                - frame_detections: List of FrameDetection objects (frame_number, detected_objects)
                - target_object_found: True if target was found, False otherwise
                - frames_with_target: List of frame numbers containing the target
                - get_image_paths_with_target(): Returns image paths for VLM processing
        
        """
        logger.log_info(self.get_class_name(), 
            f'Starting advanced scan with detection. Target: {target_object or "all objects"}')
        
        result = self.navigation_engine.scan_with_detection(
            target_object=target_object,
            yolo_model_path=yolo_model_path,
            yolo_conf_threshold=yolo_conf_threshold,
            yolo_iou_threshold=yolo_iou_threshold
        )
        
        if result.target_object_found:
            logger.log_success(self.get_class_name(), 
                f'Target "{target_object}" found in {len(result.frames_with_target)} frames.')
        else:
            logger.log_info(self.get_class_name(), 
                f'Target "{target_object}" not found. Detected objects: {result.all_unique_objects}')
        
        return result
    
    def scan_with_any_detection(
        self, 
        target_objects: list,
        yolo_world_model_path: Optional[str] = None,
        yolo_conf_threshold: float = 0.025
    ) -> 'ScanResult':
        """
        Run a 360 scan with YOLO-World for open-vocabulary targets.
        
        Args:
            target_objects: List of object names to search for (e.g., ["keys", "key", "keychain"]).
                           Providing multiple variations increases detection reliability.
            yolo_world_model_path: Path to YOLO-World PyTorch model file (e.g., yolov8m-worldv2.pt).
            yolo_conf_threshold: Confidence threshold for detections (default: 0.025)
            
        Returns:
            ScanResult: Structured result containing:
                - frame_detections: List of FrameDetection objects (frame_number, detected_objects)
                - target_object_found: True if ANY target object was found, False otherwise
                - frames_with_target: List of frame numbers containing target objects
                - get_image_paths_with_target(): Returns image paths for VLM processing
        
        """
        logger.log_info(self.get_class_name(), 
            f'Starting YOLO-World scan. Looking for: {target_objects}')
        
        result = self.navigation_engine.scan_with_any_detection(
            target_objects=target_objects,
            yolo_world_model_path=yolo_world_model_path,
            yolo_conf_threshold=yolo_conf_threshold
        )
        
        if result.target_object_found:
            logger.log_success(self.get_class_name(), 
                f'Target object(s) found in {len(result.frames_with_target)} frames.')
        else:
            logger.log_info(self.get_class_name(), 
                f'Target objects {target_objects} not found at this location.')
        
        return result
    
    def prewarm_yolo_world(
        self, 
        target_objects: list,
        yolo_world_model_path: Optional[str] = None
    ) -> bool:
        """
        Load and prime YOLO-World before takeoff to avoid first-use lag in flight.
        
        Args:
            target_objects: List of object names that will be searched for
            yolo_world_model_path: Path to YOLO-World PyTorch model file
        
        Returns:
            bool: True if model was successfully pre-warmed, False otherwise
            
        """
        logger.log_info(self.get_class_name(), 
            f'Pre-warming YOLO-World model for targets: {target_objects}')
        
        result = self.navigation_engine.prewarm_yolo_world(
            target_objects=target_objects,
            yolo_world_model_path=yolo_world_model_path
        )
        
        if result:
            logger.log_success(self.get_class_name(), 'YOLO-World model pre-warmed successfully')
        else:
            logger.log_error(self.get_class_name(), 'Failed to pre-warm YOLO-World model')
        
        return result
    
    def get_drone_instance(self):
        """
        Returns the Tello drone instance.

        Returns:
            Optional[Tello]: The Tello drone instance if available, otherwise None.
        """
        
        logger.log_info(self.get_class_name(), 'Retrieving Tello drone instance.')
        
        drone = self.navigation_engine.get_drone_instance()

        logger.log_debug(self.get_class_name(), f'Drone instance retrieved: {drone is not None}.')
        
        return drone

    def takeoff(self) -> bool:
        """
        Initiates the takeoff sequence for the drone.

        Returns:
            bool: True if the takeoff was successful, False otherwise.
        """
        logger.log_info(self.get_class_name(), 'Starting takeoff operation.')

        result = self.navigation_engine.takeoff()

        logger.log_debug(self.get_class_name(), f'Takeoff operation completed with success: {result}.')
        return result

    def land(self) -> bool:
        """
        Initiates the landing sequence for the drone.

        Returns:
            bool: True if the landing was successful, False otherwise.
        """
        logger.log_info(self.get_class_name(), 'Starting landing operation.')

        result = self.navigation_engine.land()

        logger.log_debug(self.get_class_name(), f'Landing operation completed with success: {result}.')
        return result
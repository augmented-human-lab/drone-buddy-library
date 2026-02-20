from djitellopy import Tello

from dronebuddylib.utils.logger import Logger
from dronebuddylib.models.enums import ObstacleDetectionMode
import time
import cv2
import os
import numpy as np
from typing import Optional, List, Tuple
from datetime import datetime
from PIL import Image
from dataclasses import dataclass, field

logger = Logger()


class MiDaSObstacleDetector:
    """
    MiDaS ONNX-based obstacle detector for drone navigation safety.
    
    Uses depth estimation to detect obstacles in the drone's forward path.
    Designed for real-time inference during navigation playback.
    """
    
    def __init__(self, model_path: str, detection_mode: ObstacleDetectionMode = ObstacleDetectionMode.MEDIUM):
        """
        Initialize MiDaS depth estimation for obstacle detection.
        
        Args:
            model_path: Path to the MiDaS ONNX model file
            detection_mode: Sensitivity mode for obstacle detection
        """
        self.model_path = model_path
        self.detection_mode = detection_mode
        self.session = None
        self.input_name = None
        self.output_name = None
        self.model_width = 512
        self.model_height = 384
        self._initialized = False
        
        logger.log_info('MiDaSObstacleDetector', f'Initializing with model: {model_path}')
        logger.log_info('MiDaSObstacleDetector', f'Detection mode: {detection_mode.name} (threshold: {detection_mode.value})')
    
    def initialize(self) -> bool:
        """
        Load the ONNX model. Call this before using the detector.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self._initialized:
            return True
        
        try:
            import onnxruntime as ort
            
            if not os.path.exists(self.model_path):
                logger.log_error('MiDaSObstacleDetector', f'Model file not found: {self.model_path}')
                return False
            
            # Create ONNX Runtime session
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=['DmlExecutionProvider', 'CPUExecutionProvider']
            )
            
            # Get model input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Determine model input size
            model_input_shape = self.session.get_inputs()[0].shape
            if len(model_input_shape) >= 4:
                self.model_height = model_input_shape[2] if isinstance(model_input_shape[2], int) else 384
                self.model_width = model_input_shape[3] if isinstance(model_input_shape[3], int) else 512
            
            self._initialized = True
            logger.log_success('MiDaSObstacleDetector', 
                f'Model loaded successfully. Input size: {self.model_width}x{self.model_height}')
            return True
            
        except ImportError:
            logger.log_error('MiDaSObstacleDetector', 'onnxruntime not installed. Install with: pip install onnxruntime')
            return False
        except Exception as e:
            logger.log_error('MiDaSObstacleDetector', f'Failed to initialize: {e}')
            return False
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for MiDaS ONNX model.
        
        Args:
            frame: Input RGB frame from drone camera (PyAV returns RGB)
            
        Returns:
            Preprocessed numpy array ready for ONNX inference
        """
        # Frame is already RGB from PyAV/Tello, no conversion needed
        img_rgb = frame
        
        # Resize to model's expected input size
        img_resized = cv2.resize(
            img_rgb, 
            (self.model_width, self.model_height),
            interpolation=cv2.INTER_CUBIC
        )
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_normalized - mean) / std
        
        # Convert to CHW format and add batch dimension
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_chw, axis=0)
        
        return img_batch.astype(np.float32)
    
    def get_depth_map(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a frame to get depth estimation.
        
        Args:
            frame: Input RGB frame from drone camera (PyAV format)
            
        Returns:
            Tuple of (depth_colored, depth_normalized):
                - depth_colored: Colorized depth map for visualization (BGR for OpenCV)
                - depth_normalized: Normalized depth values 0-255
        """
        if not self._initialized:
            if not self.initialize():
                return None, None
        
        original_height, original_width = frame.shape[:2]
        
        # Preprocess and run inference
        input_tensor = self.preprocess_frame(frame)
        depth_prediction = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )[0]
        
        # Remove batch dimension
        depth = depth_prediction[0]
        
        # Upscale to original size
        depth = cv2.resize(
            depth,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize to 0-255
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply colormap for visualization
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)
        
        return depth_colored, depth_normalized
    
    def check_for_obstacles(self, frame: np.ndarray, center_roi_ratio: float = 0.15) -> Tuple[bool, float, np.ndarray]:
        """
        Check if there are obstacles in the center region of the frame.
        
        Uses a focused center ROI to detect obstacles in the drone's direct forward path.
        
        Args:
            frame: Input RGB frame from drone camera (PyAV format)
            center_roi_ratio: Ratio of frame center width to analyze (0.15 = center 15% of width)
            
        Returns:
            Tuple of (obstacle_detected, center_risk_value, annotated_depth_map):
                - obstacle_detected: True if obstacle detected above threshold
                - center_risk_value: 90th percentile depth in center region (0-255)
                - annotated_depth_map: Depth map with ROI visualization
        """
        depth_colored, depth_normalized = self.get_depth_map(frame)
        
        if depth_normalized is None:
            return False, 0, None
        
        h, w = depth_normalized.shape
        
        # Focus on center 40% vertically (ignore floor/ceiling more aggressively)
        # This creates a tighter vertical band in the middle of the frame
        y_start = int(h * 0.25)  # Start at 25% from top (was 20%)
        y_end = int(h * 0.75)    # End at 75% from top (was 80%)
        
        # Define center ROI for checking (center X% of width) - smaller for focused detection
        roi_half_width = int(w * center_roi_ratio / 2)
        center_x = w // 2
        roi_x1 = center_x - roi_half_width
        roi_x2 = center_x + roi_half_width
        
        # Extract only the center ROI region (both vertical and horizontal)
        center_roi = depth_normalized[y_start:y_end, roi_x1:roi_x2]
        
        # Get the maximum depth value in the center ROI
        # Higher values in depth_normalized = closer objects (after cv2.normalize)
        max_depth_in_roi = np.max(center_roi)
        mean_depth_in_roi = np.mean(center_roi)
        
        # Use the 90th percentile to be robust to noise but still catch real obstacles
        percentile_90 = np.percentile(center_roi, 90)
        
        # Get threshold directly from the enum's value
        # The enum value IS the threshold (0-255 scale)
        # Higher threshold = less sensitive (only triggers on very close objects with high depth values)
        # Lower threshold = more sensitive (triggers on farther objects with lower depth values)
        threshold = self.detection_mode.value
        
        # Use 90th percentile for detection (robust to noise)
        # Skip detection if mode is OFF (threshold = 0)
        obstacle_detected = (threshold > 0) and (percentile_90 >= threshold)
        
        # Annotate depth map with ROI visualization
        annotated = depth_colored.copy()
        
        # Draw the vertical focus region boundaries (horizontal lines)
        cv2.line(annotated, (0, y_start), (w, y_start), (255, 255, 0), 1)  # Top of focus region
        cv2.line(annotated, (0, y_end), (w, y_end), (255, 255, 0), 1)  # Bottom of focus region
        
        # Draw center ROI box (only within the vertical focus region)
        color = (0, 0, 255) if obstacle_detected else (0, 255, 0)  # Red if obstacle, green if clear
        cv2.rectangle(annotated, (roi_x1, y_start), (roi_x2, y_end), color, 3)
        
        # Add clear status text in center of image (no debug info)
        status_text = "BLOCKED" if obstacle_detected else "PATH CLEAR"
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2
        cv2.putText(annotated, status_text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        return obstacle_detected, percentile_90, annotated
    
    def set_detection_mode(self, mode: ObstacleDetectionMode):
        """Update the obstacle detection sensitivity mode."""
        self.detection_mode = mode
        logger.log_info('MiDaSObstacleDetector', 
            f'Detection mode changed to: {mode.name} (threshold: {mode.value})')


@dataclass
class DetectionResult:
    """
    Represents a single object detection from YOLO.
    
    Attributes:
        class_name: The YOLO class name of the detected object (e.g., 'cup', 'bottle')
        confidence: Detection confidence score (0.0 - 1.0)
        bbox: Bounding box coordinates [x1, y1, x2, y2]
    """
    class_name: str
    confidence: float
    bbox: List[int] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox
        }


@dataclass 
class FrameDetection:
    """
    Represents detections for a single frame during scan.
    
    Attributes:
        frame_number: The sequential frame number during the scan (1-24 for 360° scan)
        rotation_angle: The rotation angle from start position when frame was captured
        detected_objects: List of objects detected in this frame
        image_path: Path to the saved image file
        timestamp: When the frame was captured
    """
    frame_number: int
    rotation_angle: int
    detected_objects: List[DetectionResult]
    image_path: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            'frame_number': self.frame_number,
            'rotation_angle': self.rotation_angle,
            'detected_objects': [obj.to_dict() for obj in self.detected_objects],
            'image_path': self.image_path,
            'timestamp': self.timestamp
        }
    
    def has_object(self, target_object: str) -> bool:
        """Check if this frame contains the target object (exact match)."""
        target_lower = target_object.lower().strip()
        for detection in self.detected_objects:
            # Exact match only - prevents false positives like "car" matching "carrot"
            if target_lower == detection.class_name.lower().strip():
                return True
        return False


@dataclass
class ScanResult:
    """
    Complete result of a scan_surrounding operation with YOLO detection.
    
    Attributes:
        waypoint_name: Name of the waypoint where scan was performed
        frame_detections: List of FrameDetection objects (frame_number, detected_objects)
        target_object_found: Whether the target object was found in any frame
        frames_with_target: Frame numbers that contain the target object
        all_unique_objects: Set of all unique objects detected across all frames
        image_results: Original image metadata list (for backward compatibility)
        target_objects: List of target object names being searched for (used for confidence sorting)
    """
    waypoint_name: str
    frame_detections: List[FrameDetection]
    target_object_found: bool = False
    frames_with_target: List[int] = field(default_factory=list)
    all_unique_objects: List[str] = field(default_factory=list)
    image_results: List[dict] = field(default_factory=list)
    target_objects: List[str] = field(default_factory=list)  # Target object(s) being searched for
    
    def to_dict(self) -> dict:
        return {
            'waypoint_name': self.waypoint_name,
            'frame_detections': [fd.to_dict() for fd in self.frame_detections],
            'target_object_found': self.target_object_found,
            'frames_with_target': self.frames_with_target,
            'all_unique_objects': self.all_unique_objects,
            'target_objects': self.target_objects
        }
    
    def get_frames_containing_target(self) -> List[FrameDetection]:
        """Get all frame detections that contain the target object."""
        return [fd for fd in self.frame_detections if fd.frame_number in self.frames_with_target]
    
    def _get_target_confidence(self, frame_detection: FrameDetection) -> float:
        """
        Get the maximum confidence of target object detections in a frame.
        
        For standard YOLO: Only considers detections matching target_objects (filters out other COCO classes)
        For YOLO-World: All detections are targets, so considers all detections
        
        Args:
            frame_detection: The frame to get confidence from
            
        Returns:
            float: Maximum confidence of target detections, or 0.0 if none
        """
        if not frame_detection.detected_objects:
            return 0.0
        
        # If we have specific target objects, filter to only those
        if self.target_objects:
            target_lower = [t.lower().strip() for t in self.target_objects]
            target_detections = [
                det for det in frame_detection.detected_objects 
                if det.class_name.lower().strip() in target_lower
            ]
            if target_detections:
                return max(det.confidence for det in target_detections)
            return 0.0
        
        # No target filter - use max confidence of all detections (YOLO-World case)
        return max(det.confidence for det in frame_detection.detected_objects)
    
    def get_image_paths_with_target(self) -> List[str]:
        """
        Get image paths for frames containing the target object,
        sorted by highest TARGET detection confidence first.
        
        For standard YOLO: Sorts by confidence of the specific target object only
        For YOLO-World: Sorts by confidence of any detection (all are targets)
        
        Returns:
            List[str]: Image paths, with highest confidence detection first
        """
        frames = self.get_frames_containing_target()
        sorted_frames = sorted(frames, key=self._get_target_confidence, reverse=True)
        return [fd.image_path for fd in sorted_frames]
    
    def get_best_detection_frame(self) -> Optional[FrameDetection]:
        """
        Get the frame with the highest confidence TARGET detection.
        
        Returns:
            FrameDetection: The frame with highest target confidence, or None if no detections
        """
        frames = self.get_frames_containing_target()
        if not frames:
            return None
        
        return max(frames, key=self._get_target_confidence)


class YOLOOnnxDetector:
    """
    YOLO Object Detector using ONNX Runtime for fast inference.
    Integrated into TelloNavExtra for scan operations.
    """
    
    # COCO class names (80 classes) - standard YOLO classes
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize the YOLO ONNX detector.
        
        Args:
            model_path: Path to the ONNX model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = self.COCO_CLASSES
        self.session = None
        self.imgsz = 640
        
        if not os.path.exists(model_path):
            logger.log_error('YOLOOnnxDetector', f'Model file not found: {model_path}')
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")
        
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            input_shape = self.session.get_inputs()[0].shape
            self.imgsz = input_shape[2]
            logger.log_info('YOLOOnnxDetector', f'Loaded ONNX model: {model_path}, input size: {self.imgsz}x{self.imgsz}')
        except ImportError:
            logger.log_error('YOLOOnnxDetector', 'onnxruntime not installed. Install with: pip install onnxruntime')
            raise ImportError("onnxruntime not installed")
    
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Perform object detection on a frame."""
        if frame is None or frame.size == 0 or self.session is None:
            return []
        
        original_shape = frame.shape[:2]
        input_tensor, scale, pad_w, pad_h = self._preprocess(frame)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        return self._postprocess(outputs, scale, pad_w, pad_h, original_shape)
    
    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """Preprocess frame for YOLO inference."""
        original_h, original_w = frame.shape[:2]
        scale = min(self.imgsz / original_w, self.imgsz / original_h)
        new_w, new_h = int(original_w * scale), int(original_h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = (self.imgsz - new_w) // 2, (self.imgsz - new_h) // 2
        
        padded = cv2.copyMakeBorder(resized, pad_h, self.imgsz - new_h - pad_h,
                                     pad_w, self.imgsz - new_w - pad_w,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Input is already RGB format (entire pipeline uses RGB)
        # No color conversion needed
        input_tensor = np.ascontiguousarray(
            padded.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0
        )
        return input_tensor, scale, pad_w, pad_h
    
    def _postprocess(self, outputs, scale, pad_w, pad_h, original_shape) -> List[DetectionResult]:
        """Postprocess YOLO outputs to DetectionResult objects."""
        output = outputs[0]
        if len(output.shape) == 3 and output.shape[1] < output.shape[2]:
            output = np.transpose(output, (0, 2, 1))
        
        predictions = output[0]
        boxes, scores = predictions[:, :4], predictions[:, 4:]
        class_ids, confidences = np.argmax(scores, axis=1), np.max(scores, axis=1)
        
        mask = confidences > self.conf_threshold
        boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]
        
        if len(boxes) == 0:
            return []
        
        # Convert boxes and remove padding
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1, y1 = (x_center - width / 2 - pad_w) / scale, (y_center - height / 2 - pad_h) / scale
        x2, y2 = (x_center + width / 2 - pad_w) / scale, (y_center + height / 2 - pad_h) / scale
        
        original_h, original_w = original_shape
        x1, y1 = np.clip(x1, 0, original_w), np.clip(y1, 0, original_h)
        x2, y2 = np.clip(x2, 0, original_w), np.clip(y2, 0, original_h)
        
        # NMS
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), confidences.astype(np.float32).tolist(),
                                    self.conf_threshold, self.iou_threshold)
        
        if len(indices) > 0:
            indices = indices.flatten().tolist() if isinstance(indices, np.ndarray) else [i[0] for i in indices] if isinstance(indices, tuple) else indices
        else:
            indices = []
        
        detections = []
        for idx in indices:
            class_name = self.class_names[int(class_ids[idx])] if int(class_ids[idx]) < len(self.class_names) else f"class_{class_ids[idx]}"
            detections.append(DetectionResult(
                class_name=class_name,
                confidence=float(confidences[idx]),
                bbox=[int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx])]
            ))
        return detections


class TelloNavExtra:
    # Scan rotation compensation factor (degrees) - adjust if drone under/over-rotates during scans
    # Positive value = rotate more, Negative value = rotate less
    # Default: 1 (drone typically under-rotates slightly)
    SCAN_ROTATION_COMPENSATION = 2
    
    def __init__(
        self, 
        tello: Tello = None, 
        image_dir: str = None, 
        yolo_model_path: str = None,
        yolo_conf_threshold: float = 0.25,
        yolo_iou_threshold: float = 0.45,
        yolo_world_model_path: str = None,
        yolo_world_conf_threshold: float = 0.025,
        frame_read=None
    ):
        self.tello = tello
        self.image_dir = image_dir
        self.frame_read = frame_read  # Store frame_read for use in scan operations
        self.yolo_detector = None
        self.yolo_world_model = None
        self.yolo_world_model_path = yolo_world_model_path
        self.yolo_world_conf_threshold = yolo_world_conf_threshold
        self._yolo_world_current_classes: List[str] = []  # Track currently set classes for YOLO-World
        
        # Initialize YOLO detector if model path provided
        if yolo_model_path and os.path.exists(yolo_model_path):
            try:
                self.yolo_detector = YOLOOnnxDetector(
                    yolo_model_path, 
                    conf_threshold=yolo_conf_threshold,
                    iou_threshold=yolo_iou_threshold
                )
                logger.log_info('TelloNavExtra', 'YOLO detector initialized for scan operations')
            except Exception as e:
                logger.log_warning('TelloNavExtra', f'Failed to initialize YOLO detector: {e}')
                self.yolo_detector = None
        
        # Initialize YOLO-World model if path provided (lazy loading - actual load happens on first use)
        if yolo_world_model_path and os.path.exists(yolo_world_model_path):
            logger.log_info('TelloNavExtra', f'YOLO-World model path set: {yolo_world_model_path}')

    def _load_yolo_world_model(self) -> bool:
        """
        Lazy load the YOLO-World model. Called on first use of scan_with_any_detection.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if self.yolo_world_model is not None:
            return True  # Already loaded
        
        if not self.yolo_world_model_path or not os.path.exists(self.yolo_world_model_path):
            logger.log_error('TelloNavExtra', 'YOLO-World model path not set or file does not exist')
            return False
        
        try:
            from ultralytics import YOLOWorld
            self.yolo_world_model = YOLOWorld(self.yolo_world_model_path)
            logger.log_success('TelloNavExtra', f'YOLO-World model loaded: {self.yolo_world_model_path}')
            return True
        except ImportError:
            logger.log_error('TelloNavExtra', 'ultralytics not installed. Install with: pip install ultralytics')
            return False
        except Exception as e:
            logger.log_error('TelloNavExtra', f'Failed to load YOLO-World model: {e}')
            return False
    
    def prewarm_yolo_world(self, target_objects: List[str]) -> bool:
        """
        Pre-warm the YOLO-World model by loading it and setting classes BEFORE the drone takes off.
        
        YOLO-World's set_classes() operation computes text embeddings which can take 10-20+ seconds
        on first use. If this happens while the drone is flying, the Tello's built-in safety 
        timeout (no commands for ~15 seconds) may cause an automatic landing.
        
        Call this method BEFORE takeoff when you know YOLO-World will be used (i.e., when
        searching for non-COCO objects).
        
        Args:
            target_objects: List of object names that will be searched for. This should include
                           the target object and all related objects from the VLM plan.
        
        Returns:
            bool: True if model was successfully pre-warmed, False otherwise
            
        Example:
            >>> nav_extra = TelloNavExtra(yolo_world_model_path="/path/to/yolov8m-worldv2.pt")
            >>> # Pre-warm before takeoff
            >>> nav_extra.prewarm_yolo_world(["glasses", "spectacles", "eyeglasses"])
            >>> # Now the drone can take off and scan without timeout issues
        """
        logger.log_info('TelloNavExtra', f'Pre-warming YOLO-World model for targets: {target_objects}')
        
        # Step 1: Load the model (this can take several seconds)
        if not self._load_yolo_world_model():
            logger.log_error('TelloNavExtra', 'Failed to load YOLO-World model during pre-warm')
            return False
        
        # Step 2: Set classes - this "bakes" the text embeddings and can be slow
        try:
            logger.log_info('TelloNavExtra', 'Setting YOLO-World classes (computing text embeddings)...')
            self.yolo_world_model.set_classes(target_objects)
            self._yolo_world_current_classes = list(target_objects)  # Track the currently set classes
            logger.log_success('TelloNavExtra', 
                f'YOLO-World model pre-warmed successfully. Ready to detect: {target_objects}')
            return True
        except Exception as e:
            logger.log_error('TelloNavExtra', f'Failed to set YOLO-World classes during pre-warm: {e}')
            return False

    def scan(self, current_waypoint_file: str, current_waypoint: str, frame_read=None) -> list:
        """
        Scans the surrounding of the drone while doing a 360 degree rotation.
        Captures images at 30-degree intervals (12 total images) and saves them with yaw metadata.
        
        NOTE: Video stream must be started BEFORE calling this method. This method no longer
        manages streamon/streamoff to allow continuous video streaming during navigation.

        Args:
            current_waypoint_file (str): The waypoint file name.
            current_waypoint (str): The current waypoint of the drone.
            frame_read: Tello frame reader (if None, will attempt to get from self.tello)
        
        Returns:
            list: List of dictionaries containing image info:
                  [{'image_path': str, 'filename': str, 'waypoint_file': str, 'waypoint': str, 'rotation_from_start': str, 'image_number': int, 'timestamp': str, 'format': str='JPEG'}, ...]
        """
        try:
            logger.log_info('TelloNavExtra', f'Starting 360-degree scan at waypoint: {current_waypoint}')

            # Scan configuration parameters
            ROTATION_INTERVAL = 30  # degrees (12 images total for 360°)
            TOTAL_ROTATION = 360
            STABILIZATION_TIME = 0.5  # seconds to wait after rotation before capture
            
            # Setup image storage
            base_dir = self._setup_image_storage_directory(current_waypoint_file, current_waypoint)

            # Initialize scan results list and initial drone yaw
            scan_results = []
            initial_yaw = self.get_yaw()
            
            # Use provided frame_read or get from tello
            if frame_read is None:
                frame_read = self.tello.get_frame_read()
            
            if frame_read is None:
                logger.log_error('TelloNavExtra', 'Failed to get frame reader - ensure video stream is started')
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
                            self.tello.rotate_clockwise(ROTATION_INTERVAL + self.SCAN_ROTATION_COMPENSATION)
                    except Exception as e:
                        logger.log_error('TelloNavExtra', f'Failed to rotate clockwise: {e}')
                        pass # Continue to next scan
                    
                    # Increment rotation accumulator regardless of success
                    current_rotation += ROTATION_INTERVAL
            
            # NOTE: Video stream is NOT stopped here - managed by coordinator for continuous display
            
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
            # NOTE: Video stream is NOT stopped here - managed by coordinator
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
            
            # Tello's frame_read.frame returns frames in RGB format (from PyAV)
            # PIL Image.fromarray expects RGB, so no conversion needed
            # This saves the image in RGB format for consistency
            pil_image = Image.fromarray(frame)
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
        compensation = self.SCAN_ROTATION_COMPENSATION

        try: 
            # Calculate required yaw adjustment for shortest rotation path and execute it
            if current_yaw > initial_yaw:
                if turn_degree > 180 and turn_degree < 360:
                    self.tello.rotate_clockwise(360 - turn_degree + compensation)  # Shorter rotation path
                elif turn_degree <= 180 and turn_degree > 0: 
                    self.tello.rotate_counter_clockwise(turn_degree + compensation)
                else: 
                    logger.log_debug('TelloNavExtra', 'No yaw adjustment needed')
                self.tello.send_rc_control(0, 0, 0, 0)  # Stop rotation
            else: 
                if turn_degree > 180 and turn_degree < 360: 
                    self.tello.rotate_counter_clockwise(360 - turn_degree + compensation)  # Shorter rotation path
                elif turn_degree <= 180 and turn_degree > 0: 
                    self.tello.rotate_clockwise(turn_degree + compensation)
                else: 
                    logger.log_debug('TelloNavExtra', 'No yaw adjustment needed')
                self.tello.send_rc_control(0, 0, 0, 0)  # Stop rotation
            
            return True # Return True if yaw adjustment was successful
        except Exception as e:
            logger.log_error('TelloNavExtra', f'Failed to adjust yaw: {e}')
            return False # Return False if yaw adjustment failed
    
    def scan_with_detection(
        self, 
        current_waypoint_file: str, 
        current_waypoint: str, 
        target_object: Optional[str] = None
    ) -> ScanResult:
        """
        Enhanced scan with YOLO object detection integrated.
        Performs 360-degree scan and runs YOLO detection on each captured frame.
        
        This method extends the base scan functionality by:
        1. Running YOLO object detection on each captured frame
        2. Tracking which objects are detected in which frames
        3. Optionally checking for a specific target object
        4. Returning a structured ScanResult with all detection information
        
        Args:
            current_waypoint_file (str): The waypoint file name
            current_waypoint (str): The current waypoint name where scan is performed
            target_object (str, optional): Specific object to look for (e.g., "cup", "bottle").
                                          If provided, the scan will check if this object is found.
        
        Returns:
            ScanResult: Comprehensive scan result containing:
                - waypoint_name: Name of the scanned waypoint
                - frame_detections: List of FrameDetection objects with detected objects per frame
                - target_object_found: Boolean indicating if target was found
                - frames_with_target: List of frame numbers containing target object
                - all_unique_objects: List of all unique object classes detected
                - image_results: Original image metadata (for backward compatibility)
        
        Raises:
            RuntimeError: If YOLO detector is not initialized
            
        Example:
            >>> nav_extra = TelloNavExtra(tello, yolo_model_path="/path/to/yolo.onnx")
            >>> result = nav_extra.scan_with_detection("home_waypoints.json", "living_room", "cup")
            >>> if result.target_object_found:
            ...     print(f"Found cup in frames: {result.frames_with_target}")
        """
        if self.yolo_detector is None:
            logger.log_error('TelloNavExtra', 'YOLO detector not initialized. Cannot perform scan with detection.')
            raise RuntimeError(
                "YOLO detector not initialized. Provide yolo_model_path when creating TelloNavExtra instance."
            )
        
        logger.log_info('TelloNavExtra', f'Starting scan with YOLO detection at waypoint: {current_waypoint}')
        if target_object:
            logger.log_info('TelloNavExtra', f'Looking for target object: {target_object}')
        
        # Perform the standard scan to capture images (use stored frame_read if available)
        image_results = self.scan(current_waypoint_file, current_waypoint, frame_read=self.frame_read)
        
        if not image_results:
            logger.log_warning('TelloNavExtra', 'Scan returned no images. Returning empty ScanResult.')
            return ScanResult(
                waypoint_name=current_waypoint,
                frame_detections=[],
                target_object_found=False,
                frames_with_target=[],
                all_unique_objects=[],
                image_results=[],
                target_objects=[target_object] if target_object else []
            )
        
        # Run YOLO detection on each captured image
        frame_detections: List[FrameDetection] = []
        all_detected_objects: set = set()
        frames_with_target: List[int] = []
        
        for image_info in image_results:
            image_path = image_info.get('image_path')
            frame_number = image_info.get('image_number', 0)
            rotation_angle = image_info.get('rotation_from_start', 0)
            timestamp = image_info.get('timestamp', datetime.now().isoformat())
            
            if not image_path or not os.path.exists(image_path):
                logger.log_warning('TelloNavExtra', f'Image not found: {image_path}')
                continue
            
            try:
                # Load image for detection
                # cv2.imread returns BGR, convert to RGB for our all-RGB pipeline
                frame_bgr = cv2.imread(image_path)
                if frame_bgr is None:
                    logger.log_warning('TelloNavExtra', f'Failed to load image: {image_path}')
                    continue
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                # Run YOLO detection
                detections = self.yolo_detector.detect(frame)
                
                # Create FrameDetection object
                frame_detection = FrameDetection(
                    frame_number=frame_number,
                    rotation_angle=rotation_angle,
                    detected_objects=detections,
                    image_path=image_path,
                    timestamp=timestamp
                )
                frame_detections.append(frame_detection)
                
                # Track unique objects
                for det in detections:
                    all_detected_objects.add(det.class_name)
                
                # Check for target object
                if target_object and frame_detection.has_object(target_object):
                    frames_with_target.append(frame_number)
                    logger.log_success('TelloNavExtra', 
                        f'Target object "{target_object}" found in frame {frame_number} at rotation {rotation_angle}°')
                
                # Log detection summary for this frame
                if detections:
                    detected_names = [d.class_name for d in detections]
                    logger.log_info('TelloNavExtra', 
                        f'Frame {frame_number}: Detected {len(detections)} objects: {detected_names}')
                else:
                    logger.log_debug('TelloNavExtra', f'Frame {frame_number}: No objects detected')
                    
            except Exception as e:
                logger.log_error('TelloNavExtra', f'Detection failed for frame {frame_number}: {e}')
                continue
        
        # Build final result
        target_found = len(frames_with_target) > 0
        
        result = ScanResult(
            waypoint_name=current_waypoint,
            frame_detections=frame_detections,
            target_object_found=target_found,
            frames_with_target=frames_with_target,
            all_unique_objects=list(all_detected_objects),
            image_results=image_results,
            target_objects=[target_object] if target_object else []  # Store target for confidence sorting
        )
        
        # Log summary
        logger.log_success('TelloNavExtra', 
            f'Scan with detection completed at {current_waypoint}. '
            f'Detected {len(all_detected_objects)} unique object types across {len(frame_detections)} frames.')
        
        if target_object:
            if target_found:
                logger.log_success('TelloNavExtra', 
                    f'Target object "{target_object}" FOUND in {len(frames_with_target)} frame(s): {frames_with_target}')
            else:
                logger.log_info('TelloNavExtra', 
                    f'Target object "{target_object}" NOT FOUND at this waypoint.')
        
        return result
    
    def set_yolo_detector(self, model_path: str) -> bool:
        """
        Set or update the YOLO detector with a new model.
        
        Args:
            model_path (str): Path to the YOLO ONNX model file
            
        Returns:
            bool: True if detector was successfully initialized, False otherwise
        """
        try:
            self.yolo_detector = YOLOOnnxDetector(model_path)
            logger.log_success('TelloNavExtra', f'YOLO detector updated with model: {model_path}')
            return True
        except Exception as e:
            logger.log_error('TelloNavExtra', f'Failed to set YOLO detector: {e}')
            return False
    
    def scan_with_any_detection(
        self, 
        current_waypoint_file: str, 
        current_waypoint: str, 
        target_objects: List[str]
    ) -> ScanResult:
        """
        Enhanced scan with YOLO-World object detection for open-vocabulary detection.
        Performs 360-degree scan and runs YOLO-World detection on each captured frame
        for custom specified object classes.
        
        Unlike scan_with_detection() which uses a standard YOLO model with fixed 80 COCO classes,
        this method uses YOLO-World which can detect ANY object specified in target_objects.
        YOLO-World only triggers detection when the specified objects are present, so any
        detection automatically qualifies as a match (no filtering needed).
        
        Args:
            current_waypoint_file (str): The waypoint file name
            current_waypoint (str): The current waypoint name where scan is performed
            target_objects (List[str]): List of object names to search for (e.g., ["keys", "key", "keychain"]).
                                       YOLO-World will be configured to detect ONLY these objects.
        
        Returns:
            ScanResult: Comprehensive scan result containing:
                - waypoint_name: Name of the scanned waypoint
                - frame_detections: List of FrameDetection objects with detected objects per frame
                - target_object_found: Boolean indicating if ANY target objects were found
                - frames_with_target: List of frame numbers containing target objects
                - all_unique_objects: List of all unique object classes detected
                - image_results: Original image metadata (for backward compatibility)
        
        Raises:
            RuntimeError: If YOLO-World model is not initialized
            ValueError: If target_objects list is empty
            
        Example:
            >>> nav_extra = TelloNavExtra(tello, yolo_world_model_path="/path/to/yolov8m-worldv2.pt")
            >>> result = nav_extra.scan_with_any_detection(
            ...     "home_waypoints.json", 
            ...     "living_room", 
            ...     ["keys", "key", "keychain"]
            ... )
            >>> if result.target_object_found:
            ...     print(f"Found keys in frames: {result.frames_with_target}")
        """
        # Validate inputs
        if not target_objects or len(target_objects) == 0:
            raise ValueError("target_objects list cannot be empty. Provide at least one object name to search for.")
        
        # Load YOLO-World model if not already loaded
        if not self._load_yolo_world_model():
            raise RuntimeError(
                "YOLO-World model not initialized. Provide yolo_world_model_path when creating TelloNavExtra instance."
            )
        
        logger.log_info('TelloNavExtra', f'Starting scan with YOLO-World detection at waypoint: {current_waypoint}')
        logger.log_info('TelloNavExtra', f'Looking for objects: {target_objects}')
        
        # Set the custom classes for YOLO-World to detect (skip if already set to same classes)
        # This is important because set_classes() can take 10-20+ seconds to compute text embeddings
        if sorted(self._yolo_world_current_classes) != sorted(target_objects):
            logger.log_info('TelloNavExtra', 'Setting YOLO-World classes (computing text embeddings)...')
            self.yolo_world_model.set_classes(target_objects)
            self._yolo_world_current_classes = list(target_objects)
            logger.log_debug('TelloNavExtra', f'YOLO-World classes set to: {target_objects}')
        else:
            logger.log_debug('TelloNavExtra', f'YOLO-World classes already set to: {target_objects} (skipping set_classes)')
        
        # Perform the standard scan to capture images (use stored frame_read if available)
        image_results = self.scan(current_waypoint_file, current_waypoint, frame_read=self.frame_read)
        
        if not image_results:
            logger.log_warning('TelloNavExtra', 'Scan returned no images. Returning empty ScanResult.')
            return ScanResult(
                waypoint_name=current_waypoint,
                frame_detections=[],
                target_object_found=False,
                frames_with_target=[],
                all_unique_objects=[],
                image_results=[],
                target_objects=list(target_objects)
            )
        
        # Run YOLO-World detection on each captured image
        # YOLO-World is slower than YOLO ONNX, so we need to send keepalive commands
        # to prevent the drone's 15-second inactivity timeout
        frame_detections: List[FrameDetection] = []
        all_detected_objects: set = set()
        frames_with_target: List[int] = []
        last_keepalive = time.time()
        KEEPALIVE_INTERVAL = 5  # Send keepalive every 5 seconds to stay under 15s timeout
        
        for image_info in image_results:
            image_path = image_info.get('image_path')
            frame_number = image_info.get('image_number', 0)
            rotation_angle = image_info.get('rotation_from_start', 0)
            timestamp = image_info.get('timestamp', datetime.now().isoformat())
            
            if not image_path or not os.path.exists(image_path):
                logger.log_warning('TelloNavExtra', f'Image not found: {image_path}')
                continue
            
            # Send keepalive to prevent drone timeout during slow YOLO-World inference
            if self.tello is not None and (time.time() - last_keepalive) > KEEPALIVE_INTERVAL:
                try:
                    self.tello.send_command_with_return("battery?", timeout=3)
                    logger.log_debug('TelloNavExtra', f'Sent keepalive during YOLO-World detection (frame {frame_number})')
                    last_keepalive = time.time()
                except Exception as e:
                    logger.log_warning('TelloNavExtra', f'Keepalive failed: {e}')
            
            try:
                # Run YOLO-World inference on the image
                # YOLO-World with ultralytics returns results object
                results = self.yolo_world_model(image_path, conf=self.yolo_world_conf_threshold, verbose=False)
                
                # Parse detections from YOLO-World results
                detections: List[DetectionResult] = []
                
                if results and len(results) > 0:
                    result = results[0]  # Get first result (single image)
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes
                        
                        for i in range(len(boxes)):
                            # Get class name, confidence, and bbox
                            class_id = int(boxes.cls[i].item())
                            confidence = float(boxes.conf[i].item())
                            bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                            
                            # Get class name from the model's names dict
                            class_name = result.names.get(class_id, f"class_{class_id}")
                            
                            detections.append(DetectionResult(
                                class_name=class_name,
                                confidence=confidence,
                                bbox=bbox
                            ))
                
                # Create FrameDetection object
                frame_detection = FrameDetection(
                    frame_number=frame_number,
                    rotation_angle=rotation_angle,
                    detected_objects=detections,
                    image_path=image_path,
                    timestamp=timestamp
                )
                frame_detections.append(frame_detection)
                
                # Track unique objects and frames with detections
                # With YOLO-World, ANY detection is a match since we only look for target objects
                if detections:
                    frames_with_target.append(frame_number)
                    for det in detections:
                        all_detected_objects.add(det.class_name)
                    
                    # Enhanced logging with confidence and image path for debugging
                    detection_details = [f"{d.class_name}({d.confidence:.3f})" for d in detections]
                    logger.log_success('TelloNavExtra', 
                        f'Frame {frame_number} (rotation {rotation_angle}°): Found {len(detections)} target object(s): {detection_details}')
                    logger.log_debug('TelloNavExtra', f'  -> Image path: {image_path}')
                else:
                    logger.log_debug('TelloNavExtra', f'Frame {frame_number}: No target objects detected')
                    
            except Exception as e:
                logger.log_error('TelloNavExtra', f'YOLO-World detection failed for frame {frame_number}: {e}')
                continue
        
        # Build final result
        target_found = len(frames_with_target) > 0
        
        result = ScanResult(
            waypoint_name=current_waypoint,
            frame_detections=frame_detections,
            target_object_found=target_found,
            frames_with_target=frames_with_target,
            all_unique_objects=list(all_detected_objects),
            image_results=image_results,
            target_objects=list(target_objects)  # Store targets for confidence sorting
        )
        
        # Log summary
        logger.log_success('TelloNavExtra', 
            f'Scan with YOLO-World detection completed at {current_waypoint}. '
            f'Searched for: {target_objects}. Found in {len(frames_with_target)} frame(s).')
        
        if target_found:
            logger.log_success('TelloNavExtra', 
                f'Target object(s) FOUND in frame(s): {frames_with_target}. '
                f'Detected: {list(all_detected_objects)}')
        else:
            logger.log_info('TelloNavExtra', 
                f'Target object(s) {target_objects} NOT FOUND at this waypoint.')
        
        return result
    
    def set_yolo_world_model(self, model_path: str) -> bool:
        """
        Set or update the YOLO-World model with a new model file.
        
        Args:
            model_path (str): Path to the YOLO-World PyTorch model file (e.g., yolov8m-worldv2.pt)
            
        Returns:
            bool: True if model was successfully loaded, False otherwise
        """
        if not os.path.exists(model_path):
            logger.log_error('TelloNavExtra', f'YOLO-World model file not found: {model_path}')
            return False
        
        try:
            from ultralytics import YOLOWorld
            self.yolo_world_model = YOLOWorld(model_path)
            self.yolo_world_model_path = model_path
            logger.log_success('TelloNavExtra', f'YOLO-World model updated: {model_path}')
            return True
        except ImportError:
            logger.log_error('TelloNavExtra', 'ultralytics not installed. Install with: pip install ultralytics')
            return False
        except Exception as e:
            logger.log_error('TelloNavExtra', f'Failed to set YOLO-World model: {e}')
            return False
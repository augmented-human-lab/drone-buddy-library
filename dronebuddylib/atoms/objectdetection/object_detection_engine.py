from dronebuddylib.atoms.objectdetection.vision_configs import VisionConfigs
from dronebuddylib.atoms.objectdetection.vision_engine import YoloEngine
from dronebuddylib.utils.enums import VisionAlgorithm


def detect_objects(algorithm: VisionAlgorithm, vision_config: VisionConfigs, frame):
    """
    Detects objects in a given frame using the specified vision algorithm.

    Parameters:
    - algorithm (VisionAlgorithm): The vision algorithm to be used for object detection.
    - vision_config (VisionConfigs): Configuration for the vision algorithm, including weights path.
    - frame: The input frame for which objects need to be detected.

    Returns:
    - list: List of detected objects if using YOLO V8.

    Note:
    Only YOLO V8 is implemented as of now.
    """
    if algorithm == VisionAlgorithm.YOLO_V8:
        yolo_engine = YoloEngine(vision_config.weights_path)
        yolo_engine.init_engine()
        return yolo_engine.get_object_list(frame)
    if algorithm == VisionAlgorithm.GOOGLE_VISION:
        print("Not implemented yet")
        pass


def get_bounding_boxes(algorithm: VisionAlgorithm, vision_config: VisionConfigs, frame):
    """
    Retrieves bounding boxes for objects in a given frame using the specified vision algorithm.

    Parameters:
    - algorithm (VisionAlgorithm): The vision algorithm to be used for retrieving bounding boxes.
    - vision_config (VisionConfigs): Configuration for the vision algorithm, including weights path.
    - frame: The input frame for which bounding boxes are to be retrieved.

    Returns:
    - list: List of bounding boxes if using YOLO V8.

    Note:
    Only YOLO V8 is implemented as of now.
    """
    if algorithm == VisionAlgorithm.YOLO_V8:
        yolo_engine = YoloEngine(vision_config.weights_path)
        yolo_engine.init_engine()
        return yolo_engine.get_bounding_box(frame)

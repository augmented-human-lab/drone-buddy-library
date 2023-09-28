from offline.atoms.objectdetection import VisionConfigs
from offline.atoms.objectdetection.VisionEngine import YoloEngine
from utils.enums import VisionAlgorithm


def detect_objects(algorithm: VisionAlgorithm, vision_config: VisionConfigs, frame):
    if algorithm == VisionAlgorithm.YOLO_V8:
        yolo_engine = YoloEngine(vision_config.weights_path)
        yolo_engine.init_engine()
        return yolo_engine.get_object_list(frame)
    if algorithm == VisionAlgorithm.GOOGLE_VISION:
        print("Not implemented yet")
        pass


def get_bounding_boxes(algorithm: VisionAlgorithm, vision_config: VisionConfigs, frame):
    if algorithm == VisionAlgorithm.YOLO_V8:
        yolo_engine = YoloEngine(vision_config.weights_path)
        yolo_engine.init_engine()
        return yolo_engine.get_bounding_box(frame)

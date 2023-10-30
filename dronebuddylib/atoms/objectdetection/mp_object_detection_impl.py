import pkg_resources
from mediapipe.tasks.python.vision import ObjectDetectorOptions

from dronebuddylib.atoms.objectdetection.i_object_detection import IObjectDetection

import mediapipe as mp

from dronebuddylib.models.enums import Configurations
from dronebuddylib.models.object_detected import ObjectDetected, BoundingBox, ObjectDetectionResult
from dronebuddylib.utils.utils import config_validity_check

from mediapipe.tasks.python import vision, BaseOptions

from dronebuddylib.models.engine_configurations import EngineConfigurations

VisionRunningMode = mp.tasks.vision.RunningMode


class MPObjectDetectionImpl(IObjectDetection):
    def get_class_name(self) -> str:
        return 'OBJECT_DETECTION_MP'

    def get_algorithm_name(self) -> str:
        return 'Mediapipe Object Detection'

    def __init__(self, engine_configurations: EngineConfigurations):
        config_validity_check(self.get_required_params(),
                              engine_configurations.get_configurations_for_engine(self.get_class_name()),
                              self.get_algorithm_name())

        path = pkg_resources.resource_filename(__name__, "resources/efficientdet_lite0.tflite")
        configs = engine_configurations.get_configurations_for_engine(self.get_class_name())
        model_path = configs.get(Configurations.OBJECT_DETECTION_MP_MODELS_PATH)
        if model_path is not None:
            path = model_path

        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=path),
            max_results=5,
            running_mode=VisionRunningMode.IMAGE)
        detector = vision.ObjectDetector.create_from_options(options)
        self.detector = detector

    def get_detected_objects(self, image) -> ObjectDetectionResult:
        detection_result = self.detector.detect(image)
        return_list = []
        simple_list = []
        for detected_object in detection_result.detections:
            formatted_object = ObjectDetected([], BoundingBox(detected_object.bounding_box.origin_x,
                                                              detected_object.bounding_box.origin_y,
                                                              detected_object.bounding_box.width,
                                                              detected_object.bounding_box.height))
            highest_confidence = 0
            highest_confident_category = ""
            for label in detected_object.categories:
                formatted_object.add_category(label.category_name, label.score)
                if label.score > highest_confidence:
                    highest_confidence = label.score
                    formatted_object.add_category(label.category_name, label.score)
                    highest_confident_category = label.category_name

            simple_list.append(highest_confident_category)
            return_list.append(formatted_object)

        return ObjectDetectionResult(simple_list, return_list)

    def get_bounding_boxes_of_detected_objects(self, image) -> list:
        pass

    def get_required_params(self) -> list:
        return []

    def get_optional_params(self) -> list:
        return [Configurations.OBJECT_DETECTION_MP_MODELS_PATH]

class BoundingBox:
    def __init__(self, origin_x, origin_y, width, height):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height


class DetectedCategories:
    def __init__(self, category_name: str, confidence: float):
        self.category_name = category_name
        self.confidence = confidence


class DetectedObject:
    def __init__(self, detected_categories: list[DetectedCategories], bounding_box: BoundingBox):
        self.detected_categories = detected_categories
        self.bounding_box = bounding_box

    def add_category(self, category_name: str, confidence: float):
        self.detected_categories.append(DetectedCategories(category_name, confidence))


class ObjectDetectionResult:
    def __init__(self, object_names: list, detected_objects: list[DetectedObject]):
        self.object_names = object_names
        self.detected_objects = detected_objects

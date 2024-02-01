class BoundingBox:
    def __init__(self, origin_x, origin_y, width, height):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height

    def to_json(self):
        return {
            'origin_x': self.origin_x,
            'origin_y': self.origin_y,
            'width': self.width,
            'height': self.height
        }


class DetectedCategories:
    def __init__(self, category_name: str, confidence: float):
        self.category_name = category_name
        self.confidence = confidence

    def to_json(self):
        return {
            'category_name': self.category_name,
            'confidence': self.confidence
        }


class DetectedObject:
    def __init__(self, detected_categories: list[DetectedCategories], bounding_box: BoundingBox):
        self.detected_categories = detected_categories
        self.bounding_box = bounding_box

    def add_category(self, category_name: str, confidence: float):
        self.detected_categories.append(DetectedCategories(category_name, confidence))

    def to_json(self):
        return {
            'detected_categories': [category.to_json() for category in self.detected_categories],
            'bounding_box': self.bounding_box.to_json()
        }


class ObjectDetectionResult:
    def __init__(self, object_names: list, detected_objects: list[DetectedObject]):
        self.object_names = object_names
        self.detected_objects = detected_objects

    def to_json(self):
        return {
            'object_names': self.object_names,
            'detected_objects': [detected_object.to_json() for detected_object in self.detected_objects]
        }

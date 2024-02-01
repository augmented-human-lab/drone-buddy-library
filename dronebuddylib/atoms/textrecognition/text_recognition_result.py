class Vertices:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to_json(self):
        return {
            'x': self.x,
            'y': self.y
        }


class BoundingPoly:
    def __init__(self, vertices: list[Vertices]):
        self.vertices = vertices

    def to_json(self):
        return {
            'vertices': self.vertices
        }


class TextRecognitionFullInformation:
    def __init__(self, locale, description, bounding_poly: list[Vertices]):
        self.locale = locale
        self.description = description
        self.bounding_poly = bounding_poly

    def to_json(self):
        return {
            'locale': self.locale,
            'description': self.description,
            'bounding_poly': self.bounding_poly
        }


class TextRecognitionResult:
    def __init__(self, text, locale, full_information: list[TextRecognitionFullInformation]):
        self.text = text
        self.locale = locale
        self.full_information = full_information

    def to_json(self):
        return {
            'text': self.text,
            'locale': self.locale,
            'full_information': self.full_information
        }

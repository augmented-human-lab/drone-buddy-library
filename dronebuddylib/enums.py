import enum


class ObjectDetectionReturnTypes(enum.Enum):
    """Enum for the return types of the object detection functions."""
    # The object detection function returns a list of objects.
    LABELS = 0
    # The object detection function returns a dictionary of objects.
    BBOX = 1

    CONF = 2

    ALL = 3
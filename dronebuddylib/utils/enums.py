import enum


class ObjectDetectionReturnTypes(enum.Enum):
    """Enum for the return types of the object detection functions."""
    # The object detection function returns a list of objects.
    LABELS = "LABELS"
    # The object detection function returns a dictionary of objects.
    BBOX = "BBOX"

    CONF = "CONF"

    ALL = "ALL"


class DroneCommands(enum.Enum):
    NONE = None
    TAKE_OFF = "TAKE_OFF",
    LAND = "LAND",
    FORWARD = "FORWARD",
    BACKWARD = "BACKWARD",
    LEFT = "LEFT",
    RIGHT = "RIGHT",
    UP = "UP",
    DOWN = "DOWN",
    ROTATE_CLOCKWISE = "ROTATE_CLOCKWISE",
    ROTATE_COUNTER_CLOCKWISE = "ROTATE_COUNTER_CLOCKWISE",
    BATTERY = "BATTERY",
    SPEED = "SPEED",
    HEIGHT = "HEIGHT",
    STOP = "STOP",
    DESCRIBE = "DESCRIBE",


class Language(enum.Enum):
    ENGLISH = 'en-gb',
    FRENCH = 'FR',

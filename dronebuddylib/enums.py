import enum


class ObjectDetectionReturnTypes(enum.Enum):
    """Enum for the return types of the object detection functions."""
    # The object detection function returns a list of objects.
    LABELS = 0
    # The object detection function returns a dictionary of objects.
    BBOX = 1

    CONF = 2

    ALL = 3


class DroneCommands(enum.Enum):
    NONE = None
    TAKEOFF = 0,
    LAND = 1,
    FORWARD = 2,
    BACKWARD = 3,
    LEFT = 4,
    RIGHT = 5,
    UP = 6,
    DOWN = 7,
    ROTATE_CLOCKWISE = 8,
    ROTATE_COUNTER_CLOCKWISE = 9,
    BATTERY = 10,
    SPEED = 11,
    HEIGHT = 12,
    STOP = 13,


class Language(enum.Enum):
    ENGLISH = 'en-gb',
    FRENCH = 'FR',


class DroneBuddyEngineTypes(enum.Enum):
    SPEECH_TO_TEXT = 0,
    TEXT_TO_SPEECH = 1,
    OBJECT_DETECTION = 2,
    DRONE_ENGINE = 3,

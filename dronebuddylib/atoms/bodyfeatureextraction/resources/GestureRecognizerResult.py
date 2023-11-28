class GestureCategories:
    def __init__(self, index, score, category):
        self.index = index
        self.score = score
        self.category = category


class Landmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class GestureRecognizerResult:
    def __init__(self, handedness: list[GestureCategories], gestures: list[GestureCategories],
                 landmarks: list[Landmark],
                 world_landmarks: list[Landmark]):
        self.handedness = handedness
        self.gestures = gestures
        self.landmarks = landmarks
        self.world_landmarks = world_landmarks

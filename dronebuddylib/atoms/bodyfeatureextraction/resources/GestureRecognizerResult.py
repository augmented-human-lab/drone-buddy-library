class GestureCategories:
    def __init__(self, index, score, category):
        self.index = index
        self.score = score
        self.category = category

    def to_json(self):
        return {
            'index': self.index,
            'score': self.score,
            'category': self.category
        }


class Landmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_json(self):
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z
        }


class GestureRecognizerResult:
    def __init__(self, handedness: list[GestureCategories], gestures: list[GestureCategories],
                 landmarks: list[Landmark],
                 world_landmarks: list[Landmark]):
        self.handedness = handedness
        self.gestures = gestures
        self.landmarks = landmarks
        self.world_landmarks = world_landmarks

    def to_json(self):
        return {
            'handedness': [handedness.to_json() for handedness in self.handedness],
            'gestures': [gesture.to_json() for gesture in self.gestures],
            'landmarks': [landmark.to_json() for landmark in self.landmarks],
            'world_landmarks': [world_landmark.to_json() for world_landmark in self.world_landmarks]
        }

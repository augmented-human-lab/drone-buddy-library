class RecognizedObjectObject:
    def __init__(self, name: str, confidence: float):
        self.name = name
        self.confidence = confidence

    def get_name(self) -> str:
        return self.name

    def get_confidence(self) -> []:
        return self.confidence

    def __str__(self) -> str:
        return f"RecognizedObject(name={self.name}, confidence={self.confidence})"

    def __repr__(self) -> str:
        return self.__str__()


class RecognizedObjects:
    def __init__(self, most_probable_place: RecognizedObjectObject, probable_objects: list[RecognizedObjectObject]):
        self.most_likely = most_probable_place
        self.recognition_possibilities = probable_objects

    def get_most_likely(self) -> RecognizedObjectObject:
        return self.most_likely

    def get_recognized_places(self) -> list[RecognizedObjectObject]:
        return self.recognition_possibilities

    def __str__(self) -> str:
        return f"RecognizedObject(recognized_places={self.recognition_possibilities})"

    def __repr__(self) -> str:
        return self.__str__()

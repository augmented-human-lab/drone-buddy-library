class ImageDescriberResults:
    def __init__(self, object_name: str, description: str, confidence: float):
        self.object_name = object_name
        self.description = description
        self.confidence = confidence

    def get_object_name(self) -> str:
        return self.object_name

    def get_confidence(self) -> float:
        return self.confidence

    def get_description(self) -> str:
        return self.description

    def __str__(self) -> str:
        return f"ImageDescriberResults(object_name={self.object_name}, description={self.description}, confidence={self.confidence})"

    def __repr__(self) -> str:
        return self.__str__()

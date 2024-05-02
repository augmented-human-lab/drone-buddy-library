class RecognizedFaceObject:
    def __init__(self, name: str, face_location: []):
        self.name = name
        self.face_location = face_location

    def get_name(self) -> str:
        return self.name

    def get_face_location(self) -> []:
        return self.face_location

    def __str__(self) -> str:
        return f"RecognizedFaceObject(name={self.name}, face={self.face_location})"

    def __repr__(self) -> str:
        return self.__str__()


class RecognizedFaces:
    def __init__(self, recognized_faces: list[RecognizedFaceObject]):
        self.recognized_faces = recognized_faces

    def get_recognized_faces(self) -> list[RecognizedFaceObject]:
        return self.recognized_faces

    def __str__(self) -> str:
        return f"RecognizedFaces(recognized_faces={self.recognized_faces})"

    def __repr__(self) -> str:
        return self.__str__()

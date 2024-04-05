class RecognizedPlaceObject:
    def __init__(self, name: str, place_location: []):
        self.name = name
        self.place_location = place_location

    def get_name(self) -> str:
        return self.name

    def get_place_location(self) -> []:
        return self.place_location

    def __str__(self) -> str:
        return f"RecognizedPlaceObject(name={self.name}, face={self.place_location})"

    def __repr__(self) -> str:
        return self.__str__()


class RecognizedPlaces:
    def __init__(self, recognized_places: list[RecognizedPlaceObject]):
        self.recognized_places = recognized_places

    def get_recognized_places(self) -> list[RecognizedPlaceObject]:
        return self.recognized_places

    def __str__(self) -> str:
        return f"RecognizedPlaces(recognized_places={self.recognized_places})"

    def __repr__(self) -> str:
        return self.__str__()

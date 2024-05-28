class IdentifiedObjectObject:
    def __init__(self, class_name: str, object_name: str, description: str, confidence: float):
        self.class_name = class_name
        self.object_name = object_name
        self.description = description
        self.confidence = confidence

    def get_class_name(self) -> str:
        return self.class_name

    def get_confidence(self) -> []:
        return self.confidence

    def get_object_name(self) -> str:
        return self.object_name

    def get_description(self) -> str:
        return self.description

    def __str__(self) -> str:
        return f"IdentifiedObjectObject(class_name={self.class_name}, object_name={self.object_name}, description={self.description}, confidence={self.confidence})"

    def __repr__(self) -> str:
        return self.__str__()


class IdentifiedObjects:
    def __init__(self, identified_objects: list[IdentifiedObjectObject],
                 available_objects: list[IdentifiedObjectObject]):
        self.identified_objects = identified_objects
        self.available_objects = available_objects

    def get_identified_objects(self) -> list[IdentifiedObjectObject]:
        return self.identified_objects

    def get_available_objects(self) -> list[IdentifiedObjectObject]:
        return self.available_objects

    def add_identified_object(self, identified_object: IdentifiedObjectObject):
        self.identified_objects.append(identified_object)

    def add_available_object(self, available_object: IdentifiedObjectObject):
        self.available_objects.append(available_object)

    def __str__(self) -> str:
        return f"IdentifiedObjects(identified_objects={self.identified_objects}, available_objects={self.available_objects})"

    def __repr__(self) -> str:
        return self.__str__()

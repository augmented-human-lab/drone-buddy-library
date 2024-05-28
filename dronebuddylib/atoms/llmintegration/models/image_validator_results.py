#     "object_type": "type of the object",
#
#                 "is_valid": "true if the image is good enough/ false if not", "description": "description of the object",
#                 "instructions":
#                             if the area of the image needs to be higher - INCREASE,
#
#                             if the lighting needs to be improved - LIGHTING_IMPROVE,
#
#                             if the object is incomplete : INCOMPLETE,
#
#                             if the object is not in focus : NOT_FOCUSED
class ImageValidatorResults:

    def __init__(self, object_type: str, is_valid: bool, description: str, instructions: str):
        self.object_type = object_type
        self.is_valid = is_valid
        self.description = description
        self.instructions = instructions

    def get_object_type(self) -> str:
        return self.object_type

    def get_is_valid(self) -> bool:
        return self.is_valid

    def get_description(self) -> str:
        return self.description

    def get_instructions(self) -> str:
        return self.instructions

    def __str__(self) -> str:
        return f"ImageValidatorResults(object_type={self.object_type}, is_valid={self.is_valid}, description={self.description}, instructions={self.instructions})"

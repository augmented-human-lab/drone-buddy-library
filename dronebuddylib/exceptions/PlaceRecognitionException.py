class PlaceRecognitionException(Exception):
    """
    Represents an exception raised when the place recognition functions fail.
    """

    def __init__(self, message: str, error_code: int, details=None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details

    def __str__(self):
        return self.message

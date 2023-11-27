class IntentResolutionException(Exception):
    """
    Represents an exception raised when the intent resolution fails.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message
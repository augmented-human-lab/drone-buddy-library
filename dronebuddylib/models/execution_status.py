class ExecutionStatus:
    def __init__(self, origin_class: str, origin_method: str, current_action, status: str, message: str = None):
        self.status = status
        self.message = message
        self.origin_class = origin_class
        self.origin_method = origin_method
        self.current_action = current_action

    def __str__(self):
        return f"ExecutionStatus: {self.status} - {self.message}"

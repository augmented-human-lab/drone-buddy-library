class FileWritingException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class MissingConfigurationException(Exception):
    def __init__(self, algo_name:str, config_name:str):
        message = 'Missing configuration to initialize the algorithm: ' + algo_name + ' : configuration: ' + config_name
        super().__init__(message)
        self.message = message

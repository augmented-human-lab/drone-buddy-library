class GPTConfigs:

    def __init__(self, open_ai_api_key: str, open_ai_model: str, open_ai_temperature: float,
                 loger_location: str):
        self.open_ai_api_key = open_ai_api_key
        self.open_ai_model = open_ai_model
        self.open_ai_temperature = open_ai_temperature
        self.loger_location = loger_location

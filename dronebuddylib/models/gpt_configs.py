class GPTConfigs:

    def __int__(self, open_ai_api_key: str, open_ai_model: str, open_ai_temperature: float,
                open_ai_api_url: str, loger_location: str):
        self.open_ai_api_key = open_ai_api_key
        self.open_ai_model = open_ai_model
        self.open_ai_api_url = open_ai_api_url
        self.open_ai_temperature = open_ai_temperature
        self.loger_location = loger_location

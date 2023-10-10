from dronebuddylib.models.gpt_configs import GPTConfigs


class IntentConfigs:
    def __init__(self, gpt_configs: GPTConfigs, snips_configs: dict):
        self.gpt_configs = gpt_configs
        self.snips_configs = snips_configs

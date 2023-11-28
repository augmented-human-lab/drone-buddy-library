class Intent:
    def __init__(self, intent: str, entities: dict, confidence: float):
        self.intent = intent
        self.entities = entities
        self.confidence = confidence

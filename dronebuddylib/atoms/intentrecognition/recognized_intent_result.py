class RecognizedEntities:
    def __init__(self, entity_type: str, value: str):
        self.entity_type = entity_type
        self.value = value

    def to_json(self):
        return {
            'entity_type': self.entity_type,
            'value': self.value
        }


class RecognizedIntent:
    def __init__(self, intent: str, entities: list[RecognizedEntities], confidence: float, addressed_to: bool):
        self.intent = intent
        self.entities = entities
        self.confidence = confidence
        self.addressed_to = addressed_to

    def set_entities(self, entity_list):
        self.entities = entity_list

    def to_json(self):
        return {
            'intent': self.intent,
            'entities': [entity.to_json() for entity in self.entities],
            'confidence': self.confidence,
            'addressed_to': self.addressed_to
        }

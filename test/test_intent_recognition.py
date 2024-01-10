import unittest

from dronebuddylib.atoms.intentrecognition.intent_recognition_engine import IntentRecognitionEngine
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import AtomicEngineConfigurations, IntentRecognitionAlgorithm


class TestIntentRecognition(unittest.TestCase):
    def test_intent_classification_with_SNIPS(self):
        engine_configs = EngineConfigurations({})
        engine = IntentRecognitionEngine(IntentRecognitionAlgorithm.SNIPS_NLU, engine_configs)
        result = engine.recognize_intent("can you turn to your left?")
        print(result)
        self.assertEqual(result.intent, "ROTATE_COUNTER_CLOCKWISE")

    def test_intent_classification_wth_address_with_SNIPS(self):
        engine_configs = EngineConfigurations({})
        engine = IntentRecognitionEngine(IntentRecognitionAlgorithm.SNIPS_NLU, engine_configs)
        result = engine.recognize_intent("take off sammy")
        print(result)
        self.assertEqual(result.intent, "TAKE_OFF")
        self.assertEqual(result.entities[0].entity_type, "DroneName")
        self.assertEqual(result.entities[0].value, "DroneName")

    def test_object_recognition_engine_with_CHAT_GPT(self):
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_TEMPERATURE, "0.7")
        engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_MODEL, "gpt-3.5-turbo-0613")
        engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_LOGGER_LOCATION,
                                         "C:\\Users\\Public\\projects\\drone-buddy-library\\dronebuddylib\\atoms\\intentrecognition\\resources\\stats\\")
        engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_API_KEY,
                                         "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")
        engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_API_URL,
                                         "https://api.openai.com/v1/chat/completions")
        engine = IntentRecognitionEngine(IntentRecognitionAlgorithm.CHAT_GPT, engine_configs)
        result = engine.recognize_intent("find the chair")
        print(result)
        self.assertEqual(result.intent, "RECOGNIZE_OBJECTS")


if __name__ == '__main__':
    unittest.main()

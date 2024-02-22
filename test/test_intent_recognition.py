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

    def test_intent_recognition_engine_with_CHAT_GPT(self):
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_TEMPERATURE, "0.7")
        engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_MODEL,
                                         "gpt-3.5-turbo-0613")
        engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_LOGGER_LOCATION,
                                         "C:\\Users\\Public\\projects\\drone-buddy-library\\dronebuddylib\\atoms\\intentrecognition\\resources\\stats\\")
        engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_API_KEY,
                                         "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")
        engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_API_URL,
                                         "https://api.openai.com/v1/chat/completions")
        engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_SYSTEM_PROMPT,
                                         "You are a helpful assistant acting on behalf of a drone to classify intents. " +
                                         " These intents control a drone" +
                                         " When you are given a phrase always classify it into the following intents #list" +
                                         " NEVER make up a intent, always refer the intent list provided to you and always extract from it." +
                                         " If there is no intent please match to match it to the closest one " +
                                         "NEVER make up a intent, always refer the intent list provided to you and always extract from it. " +
                                         "If there is no intent please match to match it to the closest one." +
                                         "return the result in the form of the json object" +
                                         "{\"intent\":recognized_intent, \"confidence\": confidence of the result ,\"entities\"; if there are any entities associated, " +
                                         "\"addressed_to \": if the phrase is addressed to someone set as true, else false}" +
                                         "entities is a list {\"entity_type\": type of the recognized entity , \"value\": name of the entity,}")
        engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_SYSTEM_ACTIONS_PATH,
                                         "C:\\Users\\Public\\projects\\drone-buddy-library\\dronebuddylib\\atoms\\intentrecognition\\resources\\intents.txt")
        engine = IntentRecognitionEngine(IntentRecognitionAlgorithm.CHAT_GPT, engine_configs)

        result = engine.recognize_intent("what can you see?")
        print(result)
        self.assertEqual("RECOGNIZE_OBJECTS", result.intent)


if __name__ == '__main__':
    unittest.main()

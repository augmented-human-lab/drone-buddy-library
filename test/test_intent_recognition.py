# import unittest
#
# import dronebuddylib.offline.atoms as dbl
#
#
# class MyTestCase(unittest.TestCase):
#     def test_intent_classification(self):
#         engine = dbl.init_intent_recognition_engine()
#         recognized_intent = dbl.recognize_intent(engine, "who are on your left")
#         print(recognized_intent)
#         self.assertEqual(recognized_intent.get("intent").get("intentName"), 'RECOGNIZE_PEOPLE')
#         self.assertEqual(recognized_intent.get("slots")[0].get("rawValue"), 'left')
#
#     def test_intent_classification_things(self):
#         engine = dbl.init_intent_recognition_engine()
#         recognized_intent = dbl.recognize_intent(engine, "what is on your left")
#         print(recognized_intent)
#         self.assertEqual(recognized_intent.get("intent").get("intentName"), 'DESCRIBE')
#         self.assertEqual(recognized_intent.get("slots")[0].get("rawValue"), 'left')
#
#     def test_intent_classification_rotate(self):
#         engine = dbl.init_intent_recognition_engine()
#         recognized_intent = dbl.recognize_intent(engine, "can you turn to your left?")
#         print(recognized_intent)
#         self.assertEqual(recognized_intent.get("intent").get("intentName"), 'RECOGNIZE_PEOPLE')
#
#     def test_intent_classification_wth_address(self):
#         engine = dbl.init_intent_recognition_engine()
#         recognized_intent = dbl.recognize_intent(engine, " take off sammy")
#         slots = recognized_intent.get("slots")
#         slot_values = {}
#         for slot in slots:
#             slot_values[slot.get("entity")] = slot.get("rawValue")
#         bla = recognized_intent['intent']['intentName']
#         is_me = dbl.is_addressed_to_drone(recognized_intent)
#         keys = slot_values.keys()
#         print(recognized_intent)
#         print(slots)
#         print(slot_values)
#         print(keys)
#         print(slot_values.values())
#         print(is_me)
#
#     def test_get_mentioned_entities(self):
#         print("gdfggdgdfgdfgsfrgsrfgsfrgrsewgfgregedgtedgtrgrghdhhhhhhhhhhtr")
#         engine = dbl.init_intent_recognition_engine()
#         recognized_intent = dbl.recognize_intent(engine, "what bark")
#         print(recognized_intent)
#         recognized_entities = dbl.get_mentioned_entities(recognized_intent)
#         self.assertEqual(len(recognized_entities), 2)
#         self.assertEqual(recognized_entities['address'], 'buddy')
#
#     def test_get_mentioned_entities(self):
#         engine = dbl.init_intent_recognition_engine()
#         recognized_intent = dbl.recognize_intent(engine, "What can you see")
#         recognized_entities = dbl.get_mentioned_entities(recognized_intent)
#         self.assertEqual(recognized_entities, None)
#
#     def test_is_addressed_to_drone(self):
#         engine = dbl.init_intent_recognition_engine()
#         recognized_intent = dbl.recognize_intent(engine, "buddy what is on your left")
#         is_me = dbl.is_addressed_to_drone(recognized_intent)
#         self.assertEqual(is_me, True)
#
#     def test_is_addressed_to_drone_fail(self):
#         engine = dbl.init_intent_recognition_engine()
#         recognized_intent = dbl.recognize_intent(engine, "what is on your left")
#         print(recognized_intent)
#         is_me = dbl.is_addressed_to_drone(recognized_intent)
#         self.assertEqual(is_me, False)
#
#     def test_is_addressed_entity(self):
#         engine = dbl.init_intent_recognition_engine()
#         # recognized_intent = dbl.recognize_intent(engine, "samme where are you ")
#         recognized_intent = dbl.recognize_intent(engine, "hey semmi where are you ")
#         print(recognized_intent)
#
#     def test_entities(self):
#         engine = dbl.init_intent_recognition_engine()
#         recognized_intent = dbl.recognize_intent(engine, "sammy can you move 10m to your right? ")
#         print(recognized_intent)
#
#
# if __name__ == '__main__':
#     unittest.main()

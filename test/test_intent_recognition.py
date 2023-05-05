import unittest

import dronebuddylib.atoms as dbl


class MyTestCase(unittest.TestCase):
    def test_intent_classification(self):
        engine = dbl.init_intent_recognition_engine()
        recognized_intent = dbl.recognize_intent(engine, "take off")
        print(recognized_intent)
        self.assertEqual(recognized_intent.get("intent").get("intentName"), 'TAKE_OFF')


if __name__ == '__main__':
    unittest.main()

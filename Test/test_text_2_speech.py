import unittest

import dronebuddylib as dbl


class MyTestCase(unittest.TestCase):
    def test_something(self):
        engine = dbl.init_voice_engine()
        dbl.generate_speech_and_play(engine, "sorry for the loud noises")
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()

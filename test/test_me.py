import time
import unittest

import pyaudio

import dronebuddylib as dbl
from dronebuddylib import DroneCommands


class MyTestCase(unittest.TestCase):
    def test_something(self):
        print('test_something')

    def test_integration(self):
        print('test_integration')
        voice_engine = dbl.init_text_to_speech_engine()
        speech_to_text_engine = dbl.init_speech_engine('en-us')

        try:
            time.sleep(5)
            pa = pyaudio.PyAudio()
            audio_stream = pa.open(
                rate=44100,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=8192)
            print("Microphone Ready.")

            while True:
                recognized = dbl.recognize_command(speech_to_text_engine, audio_feed=audio_stream)
                if recognized == DroneCommands.TAKEOFF:
                    dbl.generate_speech_and_play(voice_engine, 'taking off')

                elif recognized == DroneCommands.LAND:
                    dbl.generate_speech_and_play(voice_engine, 'landing')

                print(recognized)

        finally:
            if audio_stream is not None:
                audio_stream.close()

            if pa is not None:
                pa.terminate()


if __name__ == '__main__':
    unittest.main()

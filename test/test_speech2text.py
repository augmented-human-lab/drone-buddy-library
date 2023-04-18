import time
import unittest

import pyaudio

import dronebuddylib as dbl


# https://github.com/alphacep/vosk-api/blob/380fa9c7c8da71520ccb550959b0f23dc6cde4fb/python/example/test_local.py
class TestSpeech2Text(unittest.TestCase):

    def test_speech_to_text(self):
        print('its running')
        model = dbl.init_speech_engine('en-us')
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
                recognized = dbl.recognize_speech(model, audio_feed=audio_stream)
                print("im done")
                print(recognized)

        finally:
            if audio_stream is not None:
                audio_stream.close()

            if pa is not None:
                pa.terminate()

    def test_speech_to_text_spotting_words(self):
        print('its running')
        model = dbl.init_speech_engine('en-us')
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
                recognized = dbl.recognize_command(model, audio_feed=audio_stream)
                print(recognized)

        finally:
            if audio_stream is not None:
                audio_stream.close()

            if pa is not None:
                pa.terminate()

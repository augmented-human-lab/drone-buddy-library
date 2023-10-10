import json
import time
import unittest

import pyaudio

import dronebuddylib.offline.atoms as dbl
import dronebuddylib.atoms as dbl_online


# https://github.com/alphacep/vosk-api/blob/380fa9c7c8da71520ccb550959b0f23dc6cde4fb/python/example/test_local.py
class TestSpeech2Text(unittest.TestCase):

    def test_speech_to_text_online(self):
        speech_engine = dbl_online.init_google_speech_engine()
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
                # convert audio_stream to bytes
                audio_bytes = audio_stream.read(8192)

                print("im done")

                recognized = dbl_online.recognize_speech(speech_engine, audio_bytes)

        finally:
            if audio_stream is not None:
                audio_stream.close()

            if pa is not None:
                pa.terminate()

    def test_speech_to_text(self):
        print('its running')
        model = dbl.init_speech_to_text_engine('en-us')
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
        speech_to_text_engine = dbl.init_speech_to_text_engine('en-us')
        mic = pyaudio.PyAudio()

        while True:
            try:
                stream = mic.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=8192)
                print("Microphone Ready.")
                stream.start_stream()
                data = stream.read(8192)
                recognized = dbl.recognize_speech(speech_to_text_engine, audio_feed=data)
                if recognized is not None:
                    raw_text = recognized.replace('\n', '')
                    json_object = json.loads(raw_text)
                    text = json_object['text']
                    listening = False
                    stream.close()
                    print(text)
            except Exception as e:
                print(e)

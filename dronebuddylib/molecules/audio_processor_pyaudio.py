import multiprocessing

import pyaudio

import dronebuddylib.atoms as dbl_atoms
from dronebuddylib.models import EngineBank
import dill as pickle


class PyAudioProcessor:

    def __init__(self, chunk: str = 8192, format: str = pyaudio.paInt16, channels: str = 1, rate: str = 44100,
                 input: str = True, frames_per_buffer: str = 8192):
        self.channels = channels
        self.format = format
        self.rate = rate
        self.chunk = chunk
        self.input = input
        self.frames_per_buffer = frames_per_buffer


def init_pyaudio(chunk: str = 8192, format: str = pyaudio.paInt16, channels: str = 1, rate: str = 44100,
                 input: str = True, frames_per_buffer: str = 8192, speech_model: str = 'en-us'):
    pyaudio_cred = PyAudioProcessor(chunk, format, channels, rate, input, frames_per_buffer)
    return pyaudio_cred


def audio_producer(pyaudio_cred, audio_queue: multiprocessing.Queue):
    audio = pyaudio.PyAudio()
    stream = audio.open(rate=pyaudio_cred.rate, channels=pyaudio_cred.channels, format=pyaudio_cred.format, input=True,
                        frames_per_buffer=pyaudio_cred.frames_per_buffer)
    print("Microphone Ready.")
    while True:
        data = stream.read(pyaudio_cred.frames_per_buffer)
        audio_queue.put(data)


def audio_consumer(audio_queue: multiprocessing.Queue, actions_function,
                   helper_engine_bank):
    speech_to_text_engine = helper_engine_bank.get_speech_to_text_engine()
    while True:
        if not audio_queue.empty():
            data = audio_queue.get()
            recognized = dbl_atoms.recognize_command(speech_to_text_engine, data)
            print(recognized)
            action_performer(recognized, actions_function, helper_engine_bank)
            # process audio data here


def action_performer(recognized: str, action_function, helper_engine_bank):
    action_function(recognized, helper_engine_bank)


def run_audio_recognition(pyaudio_credentials, actions_function, helper_engine_bank):
    audio_queue = multiprocessing.Queue()
    producer = multiprocessing.Process(target=audio_producer, args=(pyaudio_credentials, audio_queue,))
    consumer = multiprocessing.Process(target=audio_consumer, args=(audio_queue, actions_function, helper_engine_bank))
    producer.start()
    consumer.start()
    producer.join()
    consumer.join()

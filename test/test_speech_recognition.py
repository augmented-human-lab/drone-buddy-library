import json
import re
import sys
import time
import unittest

import pyaudio

from dronebuddylib import SpeechRecognitionEngine
from dronebuddylib.atoms.speechrecognition import ResumableMicrophoneStream
from dronebuddylib.atoms.speechrecognition.microphone_stream import MicrophoneStream
from dronebuddylib.models.engine_configurations import EngineConfigurations
from dronebuddylib.models.enums import AtomicEngineConfigurations
from dronebuddylib.utils.enums import SpeechRecognitionAlgorithm, SpeechRecognitionMultiAlgoAlgorithmSupportedAlgorithms
from google.cloud import speech
import speech_recognition as speech_recognition

from dronebuddylib.utils.utils import logger


# https://github.com/alphacep/vosk-api/blob/380fa9c7c8da71520ccb550959b0f23dc6cde4fb/python/example/test_local.py
class TestSpeechRecognition(unittest.TestCase):
    RATE = 16000
    CHUNK = int(RATE / 10)  # 100ms

    # Audio recording parameters
    STREAMING_LIMIT = 240000  # 4 minutes
    SAMPLE_RATE = 16000
    CHUNK_SIZE = int(SAMPLE_RATE)  # 100ms

    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"

    def test_speech_to_text_online(self):
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_TEMPERATURE, "0.7")
        engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_MODEL,
                                         "gpt-3.5-turbo-0613")

        engine = SpeechRecognitionEngine(SpeechRecognitionAlgorithm.VOSK_SPEECH_RECOGNITION, engine_configs)
        audio_stream = None
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
                result = engine.recognize_speech(audio_bytes)
                print(result.recognized_speech)
        finally:
            if audio_stream is not None:
                audio_stream.close()

            if pa is not None:
                pa.terminate()

    def test_speech_to_text_online(self):
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_TEMPERATURE, "0.7")
        engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_MODEL,
                                         "gpt-3.5-turbo-0613")

        engine = SpeechRecognitionEngine(SpeechRecognitionAlgorithm.GOOGLE_SPEECH_RECOGNITION, engine_configs)

        with MicrophoneStream(self.RATE, self.CHUNK) as stream:
            print("blaaaaaa")
            audio_generator = stream.generator()
            requests = (
                speech.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )

            responses = engine.recognize_speech(requests)
            # Now, put the transcription responses to use.
            self.listen_print_loop(responses)

    def test_speech_to_text_multi_algo(self):
        mic_instance = speech_recognition.Microphone(device_index=1)
        speech_microphone = speech_recognition.Microphone()

        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(AtomicEngineConfigurations.SPEECH_RECOGNITION_MULTI_ALGO_ALGORITHM_NAME,
                                         SpeechRecognitionMultiAlgoAlgorithmSupportedAlgorithms.GOOGLE.name)
        engine = SpeechRecognitionEngine(SpeechRecognitionAlgorithm.MULTI_ALGO_SPEECH_RECOGNITION, engine_configs)

        while True:
            with speech_microphone as source:
                print(
                    "*********************************************************************************************************")
                print("Say something...")
                print(
                    "*********************************************************************************************************")
                print(time.time())
                try:
                    result = engine.recognize_speech(source)
                    logger.log_success("TEST", "Recognized: " + result.recognized_speech)
                except speech_recognition.WaitTimeoutError:
                    engine.recognize_speech(source)
                print(time.time(), "audio out")

    def test_speech_to_text_online_stream(self):
        engine_configs = EngineConfigurations({})
        engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_TEMPERATURE, "0.7")
        engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_MODEL,
                                         "gpt-3.5-turbo-0613")

        engine = SpeechRecognitionEngine(SpeechRecognitionAlgorithm.GOOGLE_SPEECH_RECOGNITION, engine_configs)
        mic_manager = ResumableMicrophoneStream(self.SAMPLE_RATE, self.CHUNK_SIZE)
        print(mic_manager.chunk_size)
        sys.stdout.write(self.YELLOW)
        sys.stdout.write('\nListening, say "Quit" or "Exit" to stop.\n\n')
        sys.stdout.write("End (ms)       Transcript Results/Status\n")
        sys.stdout.write("=====================================================\n")

        with mic_manager as stream:
            while not stream.closed:
                sys.stdout.write(self.YELLOW)
                sys.stdout.write(
                    "\n" + str(self.STREAMING_LIMIT * stream.restart_counter) + ": NEW REQUEST\n"
                )

                stream.audio_input = []
                audio_generator = stream.generator()

                requests = (
                    speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator
                )

                responses = engine.recognize_speech(requests)

                # Now, put the transcription responses to use.
                self.listen_print_loop_for_stream(responses, stream)

                if stream.result_end_time > 0:
                    stream.final_request_end_time = stream.is_final_end_time
                stream.result_end_time = 0
                stream.last_audio_input = []
                stream.last_audio_input = stream.audio_input
                stream.audio_input = []
                stream.restart_counter = stream.restart_counter + 1

                if not stream.last_transcript_was_final:
                    sys.stdout.write("\n")
                stream.new_stream = True

    def listen_print_loop(self, responses: object) -> str:
        """Iterates through server responses and prints them.

        The responses passed is a generator that will block until a response
        is provided by the server.

        Each response may contain multiple results, and each result may contain
        multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
        print only the transcription for the top alternative of the top result.

        In this case, responses are provided for interim results as well. If the
        response is an interim one, print a line feed at the end of it, to allow
        the next result to overwrite it, until the response is a final one. For the
        final one, print a newline to preserve the finalized transcription.

        Args:
            responses: List of server responses

        Returns:
            The transcribed text.
        """
        num_chars_printed = 0
        for response in responses:
            if not response.results:
                continue

            # The `results` list is consecutive. For streaming, we only care about
            # the first result being considered, since once it's `is_final`, it
            # moves on to considering the next utterance.
            result = response.results[0]
            if not result.alternatives:
                continue

            # Display the transcription of the top alternative.
            transcript = result.alternatives[0].transcript

            # Display interim results, but with a carriage return at the end of the
            # line, so subsequent lines will overwrite them.
            #
            # If the previous result was longer than this one, we need to print
            # some extra spaces to overwrite the previous result
            overwrite_chars = " " * (num_chars_printed - len(transcript))

            if not result.is_final:
                sys.stdout.write(transcript + overwrite_chars + "\r")
                sys.stdout.flush()

                num_chars_printed = len(transcript)

            else:
                print(transcript + overwrite_chars)

                # Exit recognition if any of the transcribed phrases could be
                # one of our keywords.
                if re.search(r"\b(exit|quit)\b", transcript, re.I):
                    print("Exiting..")
                    break

                num_chars_printed = 0

            return transcript

    def get_current_time(self) -> int:
        """Return Current Time in MS.

        Returns:
            int: Current Time in MS.
        """

        return int(round(time.time() * 1000))

    def listen_print_loop_for_stream(self, responses: object, stream: object) -> object:
        """Iterates through server responses and prints them.

        The responses passed is a generator that will block until a response
        is provided by the server.

        Each response may contain multiple results, and each result may contain
        multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
        print only the transcription for the top alternative of the top result.

        In this case, responses are provided for interim results as well. If the
        response is an interim one, print a line feed at the end of it, to allow
        the next result to overwrite it, until the response is a final one. For the
        final one, print a newline to preserve the finalized transcription.

        Arg:
            responses: The responses returned from the API.
            stream: The audio stream to be processed.

        Returns:
            The transcript of the result
        """
        for response in responses:
            if self.get_current_time() - stream.start_time > self.STREAMING_LIMIT:
                stream.start_time = self.get_current_time()
                break

            if not response.results:
                continue

            result = response.results[0]

            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript

            result_seconds = 0
            result_micros = 0

            if result.result_end_time.seconds:
                result_seconds = result.result_end_time.seconds

            if result.result_end_time.microseconds:
                result_micros = result.result_end_time.microseconds

            stream.result_end_time = int((result_seconds * 1000) + (result_micros / 1000))

            corrected_time = (
                    stream.result_end_time
                    - stream.bridging_offset
                    + (self.STREAMING_LIMIT * stream.restart_counter)
            )
            # Display interim results, but with a carriage return at the end of the
            # line, so subsequent lines will overwrite them.

            if result.is_final:
                sys.stdout.write(self.GREEN)
                sys.stdout.write("\033[K")
                sys.stdout.write(str(corrected_time) + ": " + transcript + "\n")

                stream.is_final_end_time = stream.result_end_time
                stream.last_transcript_was_final = True

                # Exit recognition if any of the transcribed phrases could be
                # one of our keywords.
                if re.search(r"\b(exit|quit)\b", transcript, re.I):
                    sys.stdout.write(self.YELLOW)
                    sys.stdout.write("Exiting...\n")
                    stream.closed = True
                    break
            else:
                sys.stdout.write(self.RED)
                sys.stdout.write("\033[K")
                sys.stdout.write(str(corrected_time) + ": " + transcript + "\r")

                stream.last_transcript_was_final = False

            return transcript

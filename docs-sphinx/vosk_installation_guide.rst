VOSK Voice Recognition
========================

The official documentation for vosk can be found `here <https://alphacephei.com/vosk/>`_.

Installation
-------------

To install VOSK Integration run the following snippet, which will install the required dependencies

.. code-block::

    pip install dronebuddylib[SPEECH_RECOGNITION_VOSK]


Usage
-------------

The OpenAi integration module requires the following configurations to function

#.  SPEECH_RECOGNITION_VOSK_LANGUAGE_MODEL_PATH - This is the path to the model that you have downloaded. This is a compulsory parameter if you are using any other language. If this is not provided, the default model will be used. The default model is the english model ( vosk-model-small-en-us-0.15 ). Vosk supported languages can be found `here <https://alphacephei.com/vosk/models>`_.



Code Example
-------------

.. code-block:: python

    engine_configs = EngineConfigurations({})
    engine_configs.add_configuration(Configurations.SPEECH_RECOGNITION_VOSK_LANGUAGE_MODEL_PATH, "0.7")

    engine = SpeechToTextEngine(SpeechRecognitionAlgorithm.VOSK_SPEECH_RECOGNITION, engine_configs)
    result = engine.recognize_speech(audio_steam=data)



How to use with the mic
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import pyaudio
    from dronebuddylib.atoms.speechrecognition.speech_to_text_engine import SpeechToTextEngine
    from dronebuddylib.models.engine_configurations import EngineConfigurations
    from dronebuddylib.models.enums import Configurations, SpeechRecognitionAlgorithm

    mic = pyaudio.PyAudio()

    # initialize speech to text engine
    engine_configs = EngineConfigurations({})
    engine_configs.add_configuration(Configurations.SPEECH_RECOGNITION_VOSK_LANGUAGE_MODEL_PATH, "C:/users/project/resources/speechrecognition/vosk-model-small-en-us-0.15")

    engine = SpeechToTextEngine(SpeechRecognitionAlgorithm.VOSK_SPEECH_RECOGNITION, engine_configs)

    # this method receives the audio input from pyaudio and returns the command
    def get_command():
        listening = True
        stream = mic.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=8192)

        while listening:
            try:
                stream.start_stream()
                # chunks the audio stream to a byte stream
                data = stream.read(8192)
                recognized = engine.recognize_speech(audio_steam=data)
                if recognized is not None:
                    listening = False
                    stream.close()
                    return recognized
            except Exception as e:
                print(e)


Output
-------------

The output will be given in the following json format

.. code-block:: json

    {
            'recognized_speech': "",
            'total_billed_time': ""
    }


Where
    - recognized_speech - Text with the recognized speech
    - total_billed_time - if a paid service the billed time, but for vosk this will be empty

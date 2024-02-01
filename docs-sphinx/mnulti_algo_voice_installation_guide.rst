Multi Algorithm Recognition
========================

Built on a third party library. The official documentation for vosk can be found `here <https://pypi.org/project/SpeechRecognition/>`_.
The library performs well in multi thread environments

officially supported algorithms

- CMU Sphinx (works offline)
- Google Speech Recognition
- Google Cloud Speech API
- Wit.ai
- Microsoft Azure Speech
- Microsoft Bing Voice Recognition (Deprecated)
- Houndify API
- IBM Speech to Text
- Snowboy Hotword Detection (works offline)
- Tensorflow
- Vosk API (works offline)
- OpenAI whisper (works offline)
- Whisper API

Installation
-------------

To install Google Integration run the following snippet, which will install the required dependencies

.. code-block::

    pip install dronebuddylib[SPEECH_RECOGNITION_MULTI]


Usage
-------------

The Google integration module requires the following configurations to function

Required Configurations
^^^^^^^^^^^^^^^^^^^^^^^^
#.  SPEECH_RECOGNITION_MULTI_ALGO_ALGORITHM_NAME - The maximum number of seconds the microphone listens before timing out.

Optional Configurations
^^^^^^^^^^^^^^^^^^^^^^^^
#.  SPEECH_RECOGNITION_MULTI_ALGO_ALGO_MIC_TIMEOUT - The maximum number of seconds the microphone listens before timing out.
#.  SPEECH_RECOGNITION_MULTI_ALGO_ALGO_PHRASE_TIME_LIMIT - The maximum duration for a single phrase before cutting off.
#.  SPEECH_RECOGNITION_MULTI_ALGO_IBM_KEY - The IBM API key for using IBM speech recognition


.. code-block:: python

    engine_configs = EngineConfigurations({})
    engine_configs.add_configuration(AtomicEngineConfigurations.SPEECH_RECOGNITION_MULTI_ALGO_ALGORITHM_NAME,
                                     SpeechRecognitionMultiAlgoAlgorithmSupportedAlgorithms.GOOGLE.name)
    engine = SpeechRecognitionEngine(SpeechRecognitionAlgorithm.MULTI_ALGO_SPEECH_RECOGNITION, engine_configs)

    result = engine.recognize_speech(audio_steam=data)



How to use with the mic
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    engine_configs = EngineConfigurations({})
    engine_configs.add_configuration(AtomicEngineConfigurations.SPEECH_RECOGNITION_MULTI_ALGO_ALGORITHM_NAME,
                                     SpeechRecognitionMultiAlgoAlgorithmSupportedAlgorithms.GOOGLE.name)
    engine = SpeechRecognitionEngine(SpeechRecognitionAlgorithm.MULTI_ALGO_SPEECH_RECOGNITION, engine_configs)

     while True:

        with speech_microphone as source:

            try:
                result = engine.recognize_speech(source)
                if result.recognized_speech is not None:
                    intent = recognize_intent_gpt(intent_engine, result.recognized_speech)
                    execute_drone_functions(intent, drone_instance, face_recognition_engine, object_recognition_engine,
                                            text_recognition_engine, voice_engine)
                else:
                    logger.log_warning("TEST", "Not Recognized: voice ")

            except speech_recognition.WaitTimeoutError:
                engine.recognize_speech(source)

            time.sleep(1)  # Sleep to simulate work and prevent a tight loop


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
    - total_billed_time - if a paid service the billed time

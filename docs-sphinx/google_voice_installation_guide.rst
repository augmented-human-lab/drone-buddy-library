Google Voice Recognition
========================

The official documentation for vosk can be found `here <https://cloud.google.com/speech-to-text>`_.
Follow the steps to create the cloud console.

#.  Installation: To use Google Speech Recognition, you first need to set up the Google Cloud environment and install necessary SDKs or libraries in your development environment.

#.  API Key and Setup: Obtain an API key from Google Cloud and configure it in your application. This key is essential for authenticating and accessing Google’s speech recognition services.

#.  Audio Input and Processing: Your application should be capable of capturing audio input, which can be sent to Google’s speech recognition service. The audio data needs to be in a format compatible with Google’s system.

#.  Handling the Output: Once Google processes the audio, it returns a text transcription. This output can be used in various ways, such as command interpretation, text analysis, or as input for other systems.

#.  Customization: Google Speech Recognition allows customization for specific vocabulary or industry terms, enhancing recognition accuracy for specialized applications.

Installation
-------------

To install Google Integration run the following snippet, which will install the required dependencies

.. code-block::

    pip install dronebuddylib[SPEECH_RECOGNITION_GOOGLE]


Usage
-------------

The Google integration module requires the following configurations to function

#.  SPEECH_RECOGNITION_GOOGLE_SAMPLE_RATE_HERTZ -
#.  SPEECH_RECOGNITION_GOOGLE_LANGUAGE_CODE -
#.  SPEECH_RECOGNITION_GOOGLE_ENCODING -


.. code-block:: python

    engine_configs = EngineConfigurations({})
    engine_configs.add_configuration(Configurations.SPEECH_RECOGNITION_GOOGLE_SAMPLE_RATE_HERTZ, 44100)
    engine_configs.add_configuration(Configurations.SPEECH_RECOGNITION_GOOGLE_LANGUAGE_CODE, "en-US")
    engine_configs.add_configuration(Configurations.SPEECH_RECOGNITION_GOOGLE_ENCODING, "LINEAR16")

    engine = SpeechToTextEngine(SpeechRecognitionAlgorithm.GOOGLE_SPEECH_RECOGNITION, engine_configs)
    result = engine.recognize_speech(audio_steam=data)



How to use with the mic
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    engine_configs = EngineConfigurations({})
    engine_configs.add_configuration(Configurations.SPEECH_RECOGNITION_GOOGLE_SAMPLE_RATE_HERTZ, 44100)
    engine_configs.add_configuration(Configurations.SPEECH_RECOGNITION_GOOGLE_LANGUAGE_CODE, "en-US")
    engine_configs.add_configuration(Configurations.SPEECH_RECOGNITION_GOOGLE_ENCODING, "LINEAR16")

    engine = SpeechToTextEngine(SpeechRecognitionAlgorithm.GOOGLE_SPEECH_RECOGNITION, engine_configs)

    with sr.Microphone() as source:
        print("Listening for commands...")
        audio = recognizer.listen(source)

        try:
            # Recognize speech using Google Speech Recognition
            command = engine.recognize_speech(audio)
            print(f"Recognized command: {command}")

            # Process and execute the command
            control_function(command)
        except e:
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
    - total_billed_time - if a paid service the billed time

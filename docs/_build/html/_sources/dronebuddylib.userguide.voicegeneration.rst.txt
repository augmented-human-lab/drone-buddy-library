Supported models
==========

Pyttsx3
~~~~~~~~~~~~~~~~~~~~~~~



pyttsx3 is a Python library that provides a simple and convenient interface for performing text-to-speech synthesis. It allows you to convert text into spoken words using various speech synthesis engines available on your system.

Here's a brief explanation of pyttsx3's key features and how it works:

#. Multi-Platform Support: pyttsx3 is designed to work on multiple platforms, including Windows, macOS, and Linux, providing cross-platform compatibility for text-to-speech functionality in Python.

#. Text-to-Speech Engines: pyttsx3 supports different speech synthesis engines, allowing you to choose the one that best suits your needs. By default, it uses the SAPI5 on Windows, NSSpeechSynthesizer on macOS, and eSpeak on Linux. Additionally, pyttsx3 can be configured to work with other third-party speech synthesis engines available on your system.

#. Installation: To install pyttsx3, you can use pip, the Python package manager, by running the following command in your terminal or command prompt:

Installation
------------
.. code-block:: bash

    $ pip install pyttsx3


#. Basic Usage: Once installed, you can start using pyttsx3 in your Python scripts. The library provides a simple and consistent API for text-to-speech synthesis.
#. Additional Features: pyttsx3 offers additional functionality to customize the speech synthesis process. You can control properties such as the speech rate, volume, voice selection, and more. The library provides methods to retrieve available voices, change voice settings, and handle events like the completion of speech.

pyttsx3 provides a straightforward and user-friendly way to incorporate text-to-speech functionality into your Python applications. By leveraging pyttsx3, you can easily generate spoken output from text for various purposes, such as accessibility, interactive systems, or voice-guided applications.

How to use with DroneBuddy
------------

The text to speech functionality utilizes the offline text to speech engine pyttsx3. More information about pyttsx3 can be found here: https://pypi.org/project/pyttsx3/


.. code-block:: python

     engine_configs = EngineConfigurations({})

     engine_configs.add_configuration(Configurations.SPEECH_GENERATION_TTS_RATE, 150)
     engine_configs.add_configuration(Configurations.SPEECH_GENERATION_TTS_VOLUME, 1.0)
     engine_configs.add_configuration(Configurations.SPEECH_GENERATION_TTS_VOICE_ID, "com.apple.speech.synthesis.voice.Alex")
     engine = SpeechGenerationEngine(SpeechGenerationAlgorithm.GOOGLE_TTS_OFFLINE, engine_configs)


     objects = engine.read_phrase(image)

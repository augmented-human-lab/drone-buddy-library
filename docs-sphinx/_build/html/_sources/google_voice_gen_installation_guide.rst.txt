Pyttsx3 Voice Generation
========================


pyttsx3 is a Python library that provides a simple and convenient interface for performing text-to-speech synthesis. It allows you to convert text into spoken words using various speech synthesis engines available on your system.
The official documentation can be found `here <https://pypi.org/project/pyttsx3/>`_

Installation
-------------

To install pyttsx3 Integration, run the following snippet, which will install the required dependencies

.. code-block::

    pip install dronebuddylib[SPEECH_GENERATION]



Usage
-------------


.. code-block:: python

    engine_configs = EngineConfigurations({})
    engine = SpeechGenerationEngine(SpeechGenerationAlgorithm.GOOGLE_TTS_OFFLINE.name, engine_configs)
    result = engine.read_phrase("Read aloud phrase")




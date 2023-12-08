Supported models
==========

VOSK
~~~~~~~~~~~~~~~~~~~~~~~

General
--------------------------


The model we are using in drone buddy is VOSK, now let’s discuss about VOSKin the same points.
If you would like to learn more, more details can be found on https://alphacephei.com/vosk/.


#. Acoustic Model: Vosk starts with an acoustic model, which is trained using deep neural networks. This model analyzes the raw audio input and converts it into a sequence of acoustic features. These features represent different aspects of the sound, such as frequency and intensity.

#. Language Model: In order to convert the acoustic features into actual words, Vosk utilizes a language model. This model incorporates grammar, vocabulary, and contextual information to predict the most likely word sequence given the acoustic features. It helps improve the accuracy and intelligibility of the recognized text.

#. Speech Recognition: Vosk combines the acoustic and language models to perform speech recognition. It processes the audio input by matching the observed acoustic features with the known language patterns. This involves comparing the features against a large database of pre-recorded speech samples to find the closest match.

#. Transcription: After the speech recognition process, Vosk generates a transcription of the spoken words in the audio. This transcription is in the form of text, allowing you to process it further in your programming application.

The way drone buddy has integrated VOSK for voice recognition is as follows.

#. Installation: You need to install the Vosk library or package in your programming environment. The installation process may vary depending on the programming language you are using.

#. Model Download: Vosk requires specific pre-trained models to function properly. You'll need to download the appropriate model files for the language and domain you intend to work with. These models are usually available on the Vosk website or GitHub repository. For now the drone buddy has downloaded the english model ( vosk-model-small-en-us-0.15 ).

#. Integration: Once you have the library installed and the models downloaded, you can integrate Vosk into your programming application. This involves loading the models, capturing audio input, and passing it to Vosk for recognition.

#. Processing Results: Finally, you can process the recognized text output from Vosk according to your application's needs. This might involve storing it, performing further analysis, or using it to trigger specific actions based on voice commands.

It's important to note that Vosk is a powerful tool, but like any voice recognition system, its accuracy can be influenced by factors such as audio quality, background noise, and speaker accents. Therefore, it's a good idea to test and fine-tune the system based on your specific use case to achieve the best results.

How to use with drone buddy
---------------------------------

Required parameters
---------------------------------
.. important::
    SPEECH_RECOGNITION_VOSK_LANGUAGE_MODEL_PATH: This is the path to the model that you have downloaded. This is a compulsory parameter if you are using any other language.
    If this is not provided, the default model will be used. The default model is the english model ( vosk-model-small-en-us-0.15 ).

Initialize the voice engine

.. code-block:: python

    from dronebuddylib.atoms.speechrecognition.speech_to_text_engine import SpeechToTextEngine
    from dronebuddylib.models.engine_configurations import EngineConfigurations
    from dronebuddylib.models.enums import Configurations, SpeechRecognitionAlgorithm

    engine_configs = EngineConfigurations({})
    engine_configs.add_configuration(Configurations.SPEECH_RECOGNITION_VOSK_LANGUAGE_MODEL_PATH, "0.7")

    engine = SpeechToTextEngine(SpeechRecognitionAlgorithm.VOSK_SPEECH_RECOGNITION, engine_configs)
    result = engine.recognize_speech(audio_steam=data)

The result will be the recognized text.

Sample code where you can use the speech to text functionality to control the drone would be as follows:

.. code-block:: python

    import dronebuddylib as dbl
    import pyaudio
    from djitellopy import Tello
    from dronebuddylib.atoms.speechrecognition.speech_to_text_engine import SpeechToTextEngine
    from dronebuddylib.models.engine_configurations import EngineConfigurations
    from dronebuddylib.models.enums import Configurations, SpeechRecognitionAlgorithm

    # initiating Tello instance
    tello = Tello()
    listening = False
    tello.connect()
    tello.streamon()

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

Google Speech Recognition
~~~~~~~~~~~~~~~~~~~~~~~

General
--------------------------


Refer https://cloud.google.com/speech-to-text for more details.


We are utilizing Google Speech Recognition in our project. Let's delve into its features and integration process, similar to how we discussed VOSK. Further details are available on Google's official speech documentation page.

#. Acoustic Model: Google Speech Recognition employs an advanced acoustic model, typically based on deep learning techniques. This model processes raw audio inputs, extracting key acoustic features essential for recognizing speech patterns, such as frequency and amplitude variations.

#. Language Model: The language model in Google Speech Recognition integrates extensive vocabulary and grammar rules. It uses this linguistic knowledge to interpret the acoustic features and generate accurate word predictions, ensuring coherent and contextually relevant speech transcription.

#. Speech Recognition Engine: Google's engine combines the acoustic and language models to decode and recognize speech. It's capable of handling various accents and dialects, offering robust performance even in challenging audio conditions.

#. Transcription and Output: The final step involves transcribing the processed speech into text. Google Speech Recognition provides real-time transcription, enabling immediate text output that can be further used in applications or stored for record-keeping.

Integration in Dronebuddy
--------------------------

#. Installation: To use Google Speech Recognition, you first need to set up the Google Cloud environment and install necessary SDKs or libraries in your development environment.

#. API Key and Setup: Obtain an API key from Google Cloud and configure it in your application. This key is essential for authenticating and accessing Google's speech recognition services.

#. Audio Input and Processing: Your application should be capable of capturing audio input, which can be sent to Google's speech recognition service. The audio data needs to be in a format compatible with Google's system.

#. Handling the Output: Once Google processes the audio, it returns a text transcription. This output can be used in various ways, such as command interpretation, text analysis, or as input for other systems.

#. Customization: Google Speech Recognition allows customization for specific vocabulary or industry terms, enhancing recognition accuracy for specialized applications.

Using Google Speech Recognition for Command Control
---------------------------------------------------

Here's a sample code snippet demonstrating how you could integrate Google Speech Recognition into an application for voice-controlled operations:

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

This code captures audio through the microphone, processes it using Google Speech Recognition, and then uses the recognized command to perform an action within the application.


Important Considerations
------------------------
Google Speech Recognition, like any advanced technology, has limitations. It requires a stable internet connection for processing and may incur costs for extensive use. The system's effectiveness can also vary based on audio quality, background noise, and speaker's clarity. Therefore, it is crucial to test and calibrate the system according to your specific requirements for optimal results.

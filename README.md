# DroneBuddyLib

## Introduction

DroneBuddy lib can be used as helper library to program your own drone. this is a offline library, so you can use it
without internet connection, which is required when you are connecting with Tello drone.

The compleete documentation can be found at [Drone Buddy documentation](https://augmented-human-lab.github.io/drone-buddy-library/index.html)


# Installation Guide

## Introduction

DroneBuddy envisions empowering everyone with the ability to personally program their intelligent drones, enriching them with desired features. At its core, DroneBuddy offers a suite of fundamental building blocks, enabling users to seamlessly integrate these elements to bring their drone to flight.

Functioning as an intuitive interface, DroneBuddy simplifies complex algorithms, stripping away the intricacies to offer straightforward input-output modalities. This approach ensures that users can accomplish their objectives efficiently, without getting bogged down in technical complexities. With DroneBuddy, the focus is on user-friendliness and ease of use, making drone programming accessible and hassle-free.

## Installation

DroneBuddy behaves as any other python library. You can find the library at [https://pypi.org/project/dronebuddylib/](https://pypi.org/project/dronebuddylib/) and install using pip.

```bash
pip install dronebuddylib
```

The installation of DroneBuddy needs the following prerequisites:

1. Python 3.9 or higher
2. Compatible pip version

> **Note:**
>
> Running `pip install dronebuddylib` will only install the drone buddy library, with only the required dependencies which are:
> - requests
> - numpy
> - cython
> - setuptools
> - packaging
> - pyparsing


# Face Recognition

Face-recognition is an open-source Python library that provides face detection, face alignment, and face recognition capabilities.
The official documentation can be found [here](https://github.com/ageitgey/face_recognition).

### Installation

The face_recognition requires the following pre-requisites:
1. dlib

#### dlib Installation

To install dlib, you need to ensure that you meet the following specifications:

- **Operating System:** dlib is compatible with Windows, macOS, and Linux operating systems.
- **Python Version:** dlib works with Python 2.7 or Python 3.x versions.
- **Compiler:** You need a C++ compiler to build and install dlib. For Windows, you can use Microsoft Visual C++ (MSVC) or MinGW. On macOS, Xcode Command Line Tools are required. On Linux, the GNU C++ Compiler (g++) is typically used.
- **Dependencies:** dlib relies on a few external dependencies, including Boost and CMake. These dependencies need to be installed beforehand to successfully build dlib.

##### Windows

The official installation instructions are found [here](https://github.com/ageitgey/face_recognition/issues/175#issue-257710508).

- To install the library, first, you need to install the dlib library. Installation instructions are here:
    1. Download CMake windows installer from [here](https://cmake.org/download/).
    2. While installing CMake select "Add CMake to the system PATH" to avoid any error in the next steps.
    3. Install Visual C++, if not installed previously.

Then run the following commands to install the face_recognition:
- cmake installation
    ```bash
    pip install cmake
    ```
- dlib installation
    ```bash
    pip install dlib
    ```
- face_recognition

##### macOS Installation

macOS installation is pretty straightforward.

```bash
pip install face_recognition
```

### Usage

##### Add Faces to the Memory

In order to proceed with the face recognition, the algorithm needs encodings of the known faces. The library has a method that is specifically designed to add these faces to the memory.

```python
engine_configs = EngineConfigurations({})
image = cv2.imread('test_clear.jpg')
engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECOGNITION_EUCLIDEAN, engine_configs)
result = engine.remember_face(image, "Jane")
```

You can check if the images and names are added to the library by simply going to the location where the library is installed.

```python
venv/Lib/site-packages/dronebuddylib/atoms/resources
```

#### Recognize Faces

```python
engine_configs = EngineConfigurations({})
image = cv2.imread('test_jane.jpg')
engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECOGNITION_EUCLIDEAN, engine_configs)
result = engine.recognize_face(image)
```

## Output

The output will be a list of names, if no people are spotted in the frame empty list will be returned. If people are spotted but not recognized, 'unknown' will be added as a list item.

### Resources

- [https://pypi.org/project/face-recognition/#description](https://pypi.org/project/face-recognition/#description)
- [https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)
- [https://github.com/ageitgey/face_recognition/issues/175#issue-257710508](https://github.com/ageitgey/face_recognition/issues/175#issue-257710508)




#  Voice Generation

## Pyttsx3 Voice Generation


pyttsx3 is a Python library that provides a simple and convenient interface for performing text-to-speech synthesis. It allows you to convert text into spoken words using various speech synthesis engines available on your system.
The official documentation can be found [here](https://pypi.org/project/pyttsx3/).

### Installation

To install pyttsx3 Integration, run the following snippet, which will install the required dependencies:

```bash
pip install dronebuddylib[SPEECH_GENERATION]
```

### Usage

```python
engine_configs = EngineConfigurations({})
engine = SpeechGenerationEngine(SpeechGenerationAlgorithm.GOOGLE_TTS_OFFLINE.name, engine_configs)
result = engine.read_phrase("Read aloud phrase")
```

# Object Detection


## Mediapipe Object Detection

The official documentation for Mediapipe can be found [here](https://developers.google.com/mediapipe).

### Installation

To install Mediapipe Integration, run the following snippet, which will install the required dependencies:

```bash
pip install dronebuddylib[OBJECT_DETECTION_MP]
```

### Usage

The Mediapipe integration module requires no configurations to function.

#### Code Example

```python
engine_configs = EngineConfigurations({})
engine = MPObjectDetectionImpl(EngineConfigurations({}))
detected_objects = engine.get_detected_objects(mp_image)
```

### Output

The output will be given in the following JSON format:

```json
{
  "message": "",
  "result": {
    "object_names": [
      ""
    ],
    "detected_objects": [
      {
        "detected_categories": [
          {
            "category_name": "",
            "confidence": 0
          }
        ],
        "bounding_box": {
          "origin_x": 0,
          "origin_y": 0,
          "width": 0,
          "height": 0
        }
      }
    ]
  }
}
```


# YOLO Object Detection

The official documentation for YOLO can be found [here](https://docs.ultralytics.com/).

## Installation

To install YOLO Integration, run the following snippet, which will install the required dependencies:

```bash
pip install dronebuddylib[OBJECT_DETECTION_YOLO]
```

## Usage

The YOLO integration module requires the following configurations to function:

- **OBJECT_DETECTION_YOLO_VERSION** - This refers to the model that you want to use for detection purposes. The list of versions can be found [here](https://docs.ultralytics.com/).

### Code Example

```python
image = cv2.imread('test_image.jpg')

engine_configs = EngineConfigurations({})
engine_configs.add_configuration(Configurations.OBJECT_DETECTION_YOLO_VERSION, "yolov8n.pt")
engine = ObjectDetectionEngine(VisionAlgorithm.YOLO, engine_configs)
objects = engine.get_detected_objects(image)
```

## Output

The output will be given in the following JSON format:

```json
{
  "message": "",
  "result": {
    "object_names": [
      ""
    ],
    "detected_objects": [
      {
        "detected_categories": [
          {
            "category_name": "",
            "confidence": 0
          }
        ],
        "bounding_box": {
          "origin_x": 0,
          "origin_y": 0,
          "width": 0,
          "height": 0
        }
      }
    ]
  }
}
```

# Voice Recognition



# Multi Algorithm Recognition

Built on a third-party library. The official documentation for vosk can be found [here](https://pypi.org/project/SpeechRecognition/).
The library performs well in multi-thread environments.

## Officially Supported Algorithms

- CMU Sphinx (works offline)
- Google Speech Recognition
- Google Cloud Speech API
- Wit.ai
- Microsoft Azure Speech
- Microsoft Bing Voice Recognition (Deprecated)
- Houndify API
- IBM Speech to Text
- Snowboy Hotword Detection (works offline)
- TensorFlow
- Vosk API (works offline)
- OpenAI Whisper (works offline)
- Whisper API

## Installation

To install Google Integration, run the following snippet, which will install the required dependencies:

```bash
pip install dronebuddylib[SPEECH_RECOGNITION_MULTI]
```

## Usage

The Google integration module requires the following configurations to function:

### Required Configurations

- **SPEECH_RECOGNITION_MULTI_ALGO_ALGORITHM_NAME** - The name of the algorithm you wish to use.

### Optional Configurations

- **SPEECH_RECOGNITION_MULTI_ALGO_ALGO_MIC_TIMEOUT** - The maximum number of seconds the microphone listens before timing out.
- **SPEECH_RECOGNITION_MULTI_ALGO_ALGO_PHRASE_TIME_LIMIT** - The maximum duration for a single phrase before cutting off.
- **SPEECH_RECOGNITION_MULTI_ALGO_IBM_KEY** - The IBM API key for using IBM speech recognition.

### Code Example

```python
engine_configs = EngineConfigurations({})
engine_configs.add_configuration(AtomicEngineConfigurations.SPEECH_RECOGNITION_MULTI_ALGO_ALGORITHM_NAME,
                                 SpeechRecognitionMultiAlgoAlgorithmSupportedAlgorithms.GOOGLE.name)
engine = SpeechRecognitionEngine(SpeechRecognitionAlgorithm.MULTI_ALGO_SPEECH_RECOGNITION, engine_configs)

result = engine.recognize_speech(audio_steam=data)
```

### How to Use with the Mic

```python
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
```

## Output

The output will be given in the following JSON format:

```json
{
    "recognized_speech": "",
    "total_billed_time": ""
}
```

Where:
- **recognized_speech** - Text with the recognized speech.
- **total_billed_time** - If a paid service, the billed time.



# Google Voice Recognition

The official documentation for Google Speech-to-Text can be found [here](https://cloud.google.com/speech-to-text).
Follow the steps to create the cloud console.

## Steps for Usage

1. **Installation:** To use Google Speech Recognition, you first need to set up the Google Cloud environment and install necessary SDKs or libraries in your development environment.
2. **API Key and Setup:** Obtain an API key from Google Cloud and configure it in your application. This key is essential for authenticating and accessing Google’s speech recognition services.
3. **Audio Input and Processing:** Your application should be capable of capturing audio input, which can be sent to Google’s speech recognition service. The audio data needs to be in a format compatible with Google’s system.
4. **Handling the Output:** Once Google processes the audio, it returns a text transcription. This output can be used in various ways, such as command interpretation, text analysis, or as input for other systems.
5. **Customization:** Google Speech Recognition allows customization for specific vocabulary or industry terms, enhancing recognition accuracy for specialized applications.

## Installation

To install Google Integration, run the following snippet, which will install the required dependencies:

```bash
pip install dronebuddylib[SPEECH_RECOGNITION_GOOGLE]
```

## Usage

The Google integration module requires the following configurations to function:

- **SPEECH_RECOGNITION_GOOGLE_SAMPLE_RATE_HERTZ**
- **SPEECH_RECOGNITION_GOOGLE_LANGUAGE_CODE**
- **SPEECH_RECOGNITION_GOOGLE_ENCODING**

### Code Example

```python
engine_configs = EngineConfigurations({})
engine_configs.add_configuration(Configurations.SPEECH_RECOGNITION_GOOGLE_SAMPLE_RATE_HERTZ, 44100)
engine_configs.add_configuration(Configurations.SPEECH_RECOGNITION_GOOGLE_LANGUAGE_CODE, "en-US")
engine_configs.add_configuration(Configurations.SPEECH_RECOGNITION_GOOGLE_ENCODING, "LINEAR16")

engine = SpeechToTextEngine(SpeechRecognitionAlgorithm.GOOGLE_SPEECH_RECOGNITION, engine_configs)
result = engine.recognize_speech(audio_steam=data)
```

### How to Use with the Mic

```python
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
```

## Output

The output will be given in the following JSON format:

```json
{
    "recognized_speech": "",
    "total_billed_time": ""
}
```

Where:
- **recognized_speech** - Text with the recognized speech.
- **total_billed_time** - If a paid service, the billed time.


# VOSK Voice Recognition

The official documentation for VOSK can be found [here](https://alphacephei.com/vosk/).

## Installation

To install VOSK Integration, run the following snippet, which will install the required dependencies:

```bash
pip install dronebuddylib[SPEECH_RECOGNITION_VOSK]
```

## Usage

The VOSK integration module requires the following configurations to function:

- **SPEECH_RECOGNITION_VOSK_LANGUAGE_MODEL_PATH** - This is the path to the model that you have downloaded. This is a compulsory parameter if you are using any other language. If this is not provided, the default model will be used. The default model is the English model (vosk-model-small-en-us-0.15). VOSK supported languages can be found [here](https://alphacephei.com/vosk/models).

### Code Example

```python
engine_configs = EngineConfigurations({})
engine_configs.add_configuration(Configurations.SPEECH_RECOGNITION_VOSK_LANGUAGE_MODEL_PATH, "0.7")

engine = SpeechToTextEngine(SpeechRecognitionAlgorithm.VOSK_SPEECH_RECOGNITION, engine_configs)
result = engine.recognize_speech(audio_steam=data)
```

### How to Use with the Mic

```python
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
```

## Output

The output will be given in the following JSON format:

```json
{
    "recognized_speech": "",
    "total_billed_time": ""
}
```

Where:
- **recognized_speech** - Text with the recognized speech.
- **total_billed_time** - If a paid service, the billed time, but for VOSK this will be empty.





# Text Recognition Module Installation

Currently, DroneBuddy supports several algorithms for text recognition:

1. pyttsx3 - Offline package

To use each of these, you can customize the installation according to your needs.

## Submodules

### Google Vision Integration

For integrating Google Vision into DroneBuddy for text recognition capabilities, please follow the specific instructions outlined in the "Google Vision Integration" guide. This module allows for robust text detection and recognition functionalities leveraging Google's cloud-based vision APIs.

For detailed installation and usage instructions, refer to the separate guide dedicated to Google Vision Integration within DroneBuddy's documentation.

(Note: The actual content and commands for the "google_text_rec_installation_guide" are not provided, hence not included in this Markdown conversion.)

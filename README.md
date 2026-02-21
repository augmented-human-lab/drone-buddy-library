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
engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECC, engine_configs)
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
engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECC, engine_configs)
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

# Navigation

The navigation module provides waypoint-based navigation for DJI Tello drones with mapping, autonomous navigation, direct waypoint navigation, and YOLO-powered scan capabilities.

## Installation

To install DroneBuddy with navigation support:

```bash
pip install dronebuddylib[NAVIGATION_TELLO]
```

This will install the necessary dependencies:
- `djitellopy` - DJI Tello drone SDK (includes `opencv-python`, `pillow`, `av`, and `numpy` as dependencies)
- `setuptools`

> **Optional dependencies for advanced features:**
> - `onnxruntime` - Required for MiDaS obstacle detection and YOLO ONNX scan detection
> - `ultralytics` - Required for YOLO-World open-vocabulary scan detection

## 2D Hierarchical Waypoint System

The navigation module uses a **two-tier hierarchical waypoint architecture** (format version 2.0):

- **Super Waypoints (SWP)**: Major hub points forming the backbone of the navigation network. Connected sequentially, each holding a human-readable name (e.g. "Kitchen", "Living Room").
- **Inner Waypoints (IWP)**: Local exploration points that branch off from a parent Super Waypoint. The drone automatically returns to the parent Super Waypoint after visiting an Inner Waypoint, enabling smart routing to any other waypoint.

This architecture allows the system to navigate between any two waypoints in the map via the shortest Super Waypoint chain, even reversing movement sequences where necessary.

## Usage and Main Operations

The navigation module uses the standard DroneBuddy engine pattern with `NavigationEngine` and supports the following operations:

1. Waypoint Mapping (2D hierarchical)
2. Interactive Navigation
3. Direct Waypoint Navigation
4. Sequential Waypoint Navigation
5. 360-Degree Surrounding Scan (basic)
6. 360-Degree Scan with YOLO Detection (COCO classes)
7. 360-Degree Scan with YOLO-World Detection (open vocabulary)

as well as 3 basic operations:
1. Return drone instance currently in use by Navigation Engine
2. Drone takeoff
3. Drone landing

### Basic Navigation Engine Setup

```python
from dronebuddylib import EngineConfigurations, NavigationAlgorithm, NavigationEngine, AtomicEngineConfigurations

# Initialize navigation engine
engine_configs = EngineConfigurations({})
engine = NavigationEngine(NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT, engine_configs)
```

### Optional Engine Configurations

```python
engine_configs = EngineConfigurations({})

# Specify waypoint directory (default: current directory)
engine_configs.add_configuration(AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_DIR, "/path/to/waypoints/directory")

# Specify a specific waypoint file for navigation
engine_configs.add_configuration(AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_FILE, "my_waypoints.json")

# Mapping movement speed (cm/s, default: 30)
engine_configs.add_configuration(AtomicEngineConfigurations.NAVIGATION_TELLO_MAPPING_MOVEMENT_SPEED, 30)

# Vertical movement scaling factor (default: 1.5)
engine_configs.add_configuration(AtomicEngineConfigurations.NAVIGATION_TELLO_VERTICAL_FACTOR, 1.5)

# Image directory for scan operations
engine_configs.add_configuration(AtomicEngineConfigurations.NAVIGATION_TELLO_IMAGE_DIR, "/path/to/images/directory")

# MiDaS depth model for obstacle detection (optional)
engine_configs.add_configuration(AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_MIDAS_MODEL_PATH, "/path/to/midas.onnx")

# Obstacle detection sensitivity: OFF, LOW, MEDIUM, HIGH, VERY_HIGH (default: OFF)
from dronebuddylib.models.enums import ObstacleDetectionMode
engine_configs.add_configuration(AtomicEngineConfigurations.NAVIGATION_TELLO_WAYPOINT_OBSTACLE_DETECTION_MODE, ObstacleDetectionMode.MEDIUM)

engine = NavigationEngine(NavigationAlgorithm.NAVIGATION_TELLO_WAYPOINT, engine_configs)
```

### Obstacle Detection Modes

| Mode | Threshold | Behaviour |
|------|-----------|----------|
| `OFF` | — | Disabled (default) |
| `LOW` | 180 | Only very close obstacles |
| `MEDIUM` | 160 | Balanced indoor navigation |
| `HIGH` | 80 | Cautious, stops for medium-distance obstacles |
| `VERY_HIGH` | 30 | Maximum caution |

When enabled, the drone checks the depth map of its forward path before every forward movement. If an obstacle is detected, the drone waits up to 30 seconds for the path to clear before proceeding.

### Navigation Instructions (For Direct and Sequential Navigation)

Use the `NavigationInstruction` enum for waypoint navigation behaviour:

- `NavigationInstruction.CONTINUE` - Keep the drone flying after reaching the waypoint
- `NavigationInstruction.HALT` - Land the drone after reaching the waypoint

### Waypoint Mapping

Create 2D hierarchical waypoint maps through manual drone control:

```python
# Start mapping mode - provides real-time keyboard control interface
result = engine.map_location()
print(f"Mapping completed. Created {len(result)} waypoints.")
```

The mapping interface guides you to create **Super Waypoints** (major hub areas) and **Inner Waypoints** (local exploration points). Controls:

| Key | Action |
|-----|--------|
| `W / A / S / D` | Move forward / left / backward / right |
| `↑ / ↓` | Move up / down |
| `← / →` | Rotate counter-clockwise / clockwise |
| `X` | Mark a waypoint (prompts for Super or Inner, and name) |
| `Q` | Finish mapping and save |

### Interactive Navigation

Navigate between existing waypoints with an interactive menu:

```python
# Start interactive navigation mode - displays a waypoint selection menu
result = engine.navigate()
print(f"Navigation completed. Visited {len(result)} waypoints.")
```

### Direct Waypoint Navigation

Navigate directly to a specific waypoint by name or ID:

```python
from dronebuddylib.atoms.navigation import NavigationInstruction

# Navigate to a named waypoint and keep flying
result = engine.navigate_to_waypoint("Kitchen", NavigationInstruction.CONTINUE)

# Navigate to a waypoint and land
result = engine.navigate_to_waypoint("START", NavigationInstruction.HALT)

print(f"Landed: {result[0]}, Currently at: {result[1]}")
```

### Sequential Waypoint Navigation

Navigate through a list of waypoints in order:

```python
from dronebuddylib.atoms.navigation import NavigationInstruction

waypoints = ["Kitchen", "Living Room", "Bedroom", "START"]
result = engine.navigate_to(waypoints, NavigationInstruction.HALT)

print(f"Visited waypoints: {result}")
```

### 360-Degree Surrounding Scan (Basic)

Capture images while performing a full 360-degree rotation:

```python
images = engine.scan_surrounding()
print(f"Scan completed. Captured {len(images)} images.")
```

### 360-Degree Scan with YOLO Detection (COCO Classes)

Perform a full scan and run YOLO detection on every captured frame. Use this when searching for objects from the standard [COCO 80-class list](https://cocodataset.org/) (person, cup, laptop, bottle, etc.):

```python
result = engine.scan_with_detection(
    target_object="cup",
    yolo_model_path="models/yolov8n_640x640.onnx",
    yolo_conf_threshold=0.25,
    yolo_iou_threshold=0.45
)

if result.target_object_found:
    image_paths = result.get_image_paths_with_target()
    print(f"Found cup in {len(result.frames_with_target)} frames")
else:
    print(f"Cup not found. Detected: {result.all_unique_objects}")
```

### 360-Degree Scan with YOLO-World Detection (Open Vocabulary)

Use YOLO-World for objects **not** in the COCO 80-class list (glasses, keys, wallet, etc.). Provide the object name and synonyms to improve detection reliability:

```python
# Pre-warm YOLO-World BEFORE takeoff to avoid timeout during flight
engine.prewarm_yolo_world(
    target_objects=["glasses", "spectacles", "eyeglasses"],
    yolo_world_model_path="models/yolov8m-worldv2.pt"
)

engine.takeoff()

result = engine.scan_with_any_detection(
    target_objects=["glasses", "spectacles", "eyeglasses", "eyewear"],
    yolo_world_model_path="models/yolov8m-worldv2.pt",
    yolo_conf_threshold=0.025
)

if result.target_object_found:
    print(f"Found in {len(result.frames_with_target)} frames")
```

> **Important:** Always call `prewarm_yolo_world()` **before** `takeoff()` when using YOLO-World. Computing text embeddings on first use can take 10–20 seconds, which may trigger the Tello's automatic landing safety timeout.

## Output Format

### Mapping Results
```python
[
    {"id": "SWP_001", "name": "START"},
    {"id": "SWP_002", "name": "Kitchen"},
    {"id": "SWP_002_IWP_Counter", "name": "Counter"}
]
```

### Navigation Results
```python
["SWP_002", "SWP_003", "SWP_001", ...]  # List of waypoint IDs visited in order
```

### Direct Navigation Results
```python
[False, "SWP_002"]  # [landed_status, current_waypoint_id]
[True,  "SWP_001"]  # drone has landed
```

### Sequential Navigation Results
```python
["SWP_002", "SWP_003", "SWP_001", ...]  # List of waypoint IDs reached in sequence
```

### Basic Scan Results
```python
[
    {
        "image_path": "/path/to/image0.jpg",
        "filename": "image0.jpg",
        "waypoint": "SWP_002",
        "rotation_from_start": 0,
        "image_number": 1,
        "timestamp": "20260131_143022_123",
        "format": "JPEG"
    },
    ...
]
```

### YOLO Scan Results (`ScanResult`)

```python
result.target_object_found    # bool: True if target detected in any frame
result.frames_with_target     # List[int]: frame numbers containing the target
result.all_unique_objects     # List[str]: all unique class names detected across all frames
result.frame_detections       # List[FrameDetection]: per-frame detection detail
result.get_image_paths_with_target()  # List[str]: image file paths containing the target

# Each FrameDetection contains:
#   frame_number    int     (1–24 for a full 360° scan)
#   rotation_angle  int     (0°, 15°, 30°, … 345°)
#   detected_objects List[DetectionResult]  (class_name, confidence, bbox)
#   image_path      str
```

## Waypoint File Format (v2.0)

Waypoint files generated by the 2D hierarchical mapping system use format version 2.0:

```json
{
  "session_info": {
    "format_version": "2.0",
    "created": "2026-01-31T16:32:25"
  },
  "super_waypoints": [
    {
      "id": "SWP_001",
      "name": "START",
      "index": 0,
      "is_super_waypoint": true,
      "movements_to_here": [],
      "inner_waypoints": []
    },
    {
      "id": "SWP_002",
      "name": "Kitchen",
      "index": 1,
      "is_super_waypoint": true,
      "movements_to_here": [
        {
          "id": "4a019fcf-e595-482f-b3dc-aa129e5fc32d",
          "type": "move",
          "yaw": 91,
          "start_yaw": 0,
          "distance": 191.17
        },
        {
          "id": "1fae8501-6625-487b-8562-25b43f387a91",
          "type": "lift",
          "direction": "up",
          "distance": 52.3
        }
      ],
      "inner_waypoints": [
        {
          "id": "SWP_002_IWP_Counter",
          "name": "Counter",
          "movements_to_here": [
            {
              "id": "76cdf44c-37a5-4661-bdd7-07f87112b182",
              "type": "move",
              "yaw": 0,
              "start_yaw": 91,
              "distance": 92.63
            }
          ]
        }
      ]
    }
  ]
}
```

**Movement types:**
- `"move"` — Horizontal movement. Fields: `yaw` (target heading °), `start_yaw` (heading at start of movement), `distance` (cm).
- `"lift"` — Vertical movement. Fields: `direction` (`"up"` or `"down"`), `distance` (cm).

---

# VLM-Based Object Finder (Planner)

The Planner module provides a fully autonomous object-finding pipeline that combines a Vision-Language Model (VLM) for intelligent planning, dual YOLO detection, and 2D hierarchical navigation. Given a natural-language request such as *"Find my coffee cup"*, the system plans a search route, flies the drone to each candidate location, runs object detection, and asks the user to confirm before reporting success.

## Installation

Install the navigation dependencies first, then the VLM SDK for your chosen provider:

```bash
pip install dronebuddylib[NAVIGATION_TELLO]
pip install onnxruntime          # for YOLO ONNX and MiDaS
pip install ultralytics          # for YOLO-World (non-COCO objects)

# Install your VLM provider SDK:
pip install openai               # OpenAI (GPT-4o, GPT-5, …)
pip install anthropic            # Anthropic (Claude)
pip install google-generativeai  # Google (Gemini)
```

## Supported VLM Providers

| Provider | Models | Parameter value |
|----------|--------|-----------------|
| OpenAI | gpt-4o, gpt-4-turbo, gpt-5 | `"openai"` |
| Anthropic | claude-3-5-sonnet-20241022, claude-3-opus | `"anthropic"` |
| Google | gemini-1.5-pro, gemini-1.5-flash | `"google"` |

## Basic Usage

```python
from dronebuddylib.atoms.planning import PlannerEngine, PlannerConfigs
from dronebuddylib.models.enums import ObstacleDetectionMode

config = PlannerConfigs(
    vlm_provider="openai",
    vlm_api_key="sk-...",
    vlm_model="gpt-4o",              # optional – uses provider default if omitted
    yolo_model_path="models/yolo11m_320x320.onnx",
    yolo_world_model_path="models/yolov8m-worldv2.pt",
    midas_model_path="models/midas_small_384x288.onnx",
    obstacle_detection_mode="MEDIUM",
    waypoint_file_path="drone_movements_20260131.json",
    max_replan_attempts=2
)

engine = PlannerEngine.from_config(config)
result = engine.find_object("Find my coffee cup")

if result.success:
    print(f"Found '{result.target_object}' at {result.found_at_waypoint}")
else:
    print(f"Could not find '{result.target_object}'")
```

## Configuration Reference (`PlannerConfigs`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vlm_provider` | str | `"openai"` | VLM provider: `"openai"`, `"anthropic"`, `"google"` |
| `vlm_api_key` | str | `""` | API key for the VLM provider |
| `vlm_model` | str | provider default | Model name (e.g. `"gpt-4o"`) |
| `vlm_temperature` | float | `0.3` | Response randomness (lower = more deterministic) |
| `yolo_model_path` | str | `""` | Path to YOLO ONNX model for COCO 80-class detection |
| `yolo_confidence_threshold` | float | `0.25` | Minimum confidence for standard YOLO detections |
| `yolo_iou_threshold` | float | `0.45` | NMS IOU threshold for standard YOLO |
| `yolo_world_model_path` | str | `""` | Path to YOLO-World `.pt` model (required for non-COCO objects) |
| `yolo_world_confidence_threshold` | float | `0.025` | Confidence threshold for YOLO-World |
| `midas_model_path` | str | `""` | Path to MiDaS ONNX model for obstacle detection |
| `obstacle_detection_mode` | str | `"OFF"` | `OFF`, `LOW`, `MEDIUM`, `HIGH`, `VERY_HIGH` |
| `waypoint_file_path` | str | `""` | Path to the 2D waypoint JSON file |
| `waypoint_directory` | str | `""` | Directory containing waypoint files |
| `scan_image_directory` | str | `None` | Directory for saving scan frame images |
| `max_replan_attempts` | int | `2` | Maximum VLM re-planning attempts if object not found |

## How It Works

### 1. VLM Planning
The user's request is sent to the configured VLM alongside the list of available waypoint names. The VLM:
- Determines whether the target object is one of the **80 COCO classes** (e.g. `cup`, `laptop`) or a **custom object** (e.g. `glasses`, `keys`).
- Generates 3–5 synonym/related terms for custom objects to increase YOLO-World hit rate.
- Produces an ordered `navigate → scan` action plan, prioritising the most semantically likely locations first.

### 2. Navigation & Detection
For each `navigate → scan` pair:
- The drone navigates to the target waypoint via the 2D hierarchical pathfinding engine.
- Optional MiDaS obstacle detection pauses forward movement if the path is blocked.
- A full 360° scan is performed (24 frames × 15° rotation).
- **Standard YOLO ONNX** is used for COCO-class objects; **YOLO-World PyTorch** is used for custom objects.

### 3. Confirmation
When a match is detected the VLM analyses the best detection frame and generates a human-readable description. The user confirms or rejects the detected object.

### 4. Re-planning
If the object is not found or the user rejects the detection, the VLM generates a new plan prioritising unvisited waypoints, up to `max_replan_attempts` times.

## Session Result (`PlannerSessionResult`)

```python
result.success              # bool   – True if object was found and confirmed
result.target_object        # str    – Object that was searched for
result.found_at_waypoint    # str    – Waypoint name where object was found
result.waypoints_visited    # List[str] – All waypoints visited during the session
result.scans_performed      # int    – Number of 360° scans executed
result.object_description   # str    – VLM description of the detected object
result.session_duration     # float  – Total session time in seconds
result.final_state          # PlannerState enum value
result.error_message        # str    – Set if an error occurred
```

## GUI Mode

The planner ships with a Tkinter-based graphical interface providing a chat-like interaction panel, live video feed, and a separate log viewer:

```python
from dronebuddylib.atoms.planning.planner_gui import PlannerGUIApp
from dronebuddylib.atoms.planning import PlannerConfigs

config = PlannerConfigs(
    vlm_provider="openai",
    vlm_api_key="sk-...",
    yolo_model_path="models/yolo11m_320x320.onnx",
    waypoint_file_path="my_waypoints.json"
)

app = PlannerGUIApp(config=config)
app.run()
```

Alternatively, run the bundled example directly:

```bash
python examples/planner_example.py          # GUI mode (default)
python examples/planner_example.py --cli    # terminal mode
```

## Submodules

### Google Vision Integration

For integrating Google Vision into DroneBuddy for text recognition capabilities, please follow the specific instructions outlined in the "Google Vision Integration" guide. This module allows for robust text detection and recognition functionalities leveraging Google's cloud-based vision APIs.

For detailed installation and usage instructions, refer to the separate guide dedicated to Google Vision Integration within DroneBuddy's documentation.

(Note: The actual content and commands for the "google_text_rec_installation_guide" are not provided, hence not included in this Markdown conversion.)

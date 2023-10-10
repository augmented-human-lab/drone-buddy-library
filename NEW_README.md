
---

# `IntentRecognitionEngine` 

## Overview

The `IntentRecognitionEngine` class is a versatile engine designed for recognizing user intents. It leverages different algorithms and acts as a bridge between the user's input and the underlying intent recognition systems like ChatGPT and Snips NLU.

## Attributes

- **intent_recognizer:** 
  - Description: An instance of the intent recognition algorithm chosen.

## Constructors

### `__init__(self, algorithm: IntentRecognitionAlgorithm, config: IntentConfigs) -> None`

Initializes the `IntentRecognitionEngine` using a specific algorithm and its associated configuration.

- **Parameters:**
  - `algorithm (IntentRecognitionAlgorithm)`: The desired algorithm for intent recognition. Supported values are from the `IntentRecognitionAlgorithm` enum, including `CHAT_GPT` and `SNIPS_NLU`.
  - `config (IntentConfigs)`: Configuration parameters tailored for the selected algorithm.

### Examples:

```python
engine = IntentRecognitionEngine(IntentRecognitionAlgorithm.CHAT_GPT, config)
intent = engine.recognize_intent("Turn off the lights.")
```

## Methods

### `recognize_intent(self, text: str) -> str`

Given a user's input text, this method recognizes and returns the corresponding intent based on the configured algorithm.

- **Parameters:**
  - `text (str)`: Input text from the user for which the intent is to be recognized.
- **Returns:**
  - `str`: The recognized intent.

### Examples:

```python
engine = IntentRecognitionEngine(IntentRecognitionAlgorithm.CHAT_GPT, config)
intent = engine.recognize_intent("What's the weather today?")
# Expected output: "get_weather"
```

---

# `OfflineIntentRecognitionEngine` 

## Overview

The `OfflineIntentRecognitionEngine` class offers an offline intent recognition system leveraging Snips NLU for processing user intents. This system recognizes intents, retrieves intent names, mentioned entities, and checks if an intent is addressed to the drone.

## Attributes

- **engine (SnipsNLUEngine):** 
  - Description: The Snips NLU engine instance used for intent recognition.

## Methods

### `__init__(self, dataset_path: str = None, config: str = CONFIG_EN) -> None`

Initializes the OfflineIntentRecognitionEngine with the given dataset path and configuration.

- **Parameters:**
  - `dataset_path (str, optional)`: Path to the JSON dataset file. Defaults to None.
  - `config (str, optional)`: Configuration for the SnipsNLUEngine. Defaults to `CONFIG_EN`.

### `recognize_intent(self, text: str) -> dict`

Parses the given text and determines the intent.

- **Parameters:**
  - `text (str)`: Input string for intent recognition.
- **Returns:**
  - `dict`: Dictionary with the detected intent and associated slots.

### `get_intent_name(self, intent: dict, threshold: float = 0.5) -> str`

Extracts the intent name if it meets the threshold.

- **Parameters:**
  - `intent (dict)`: Recognized intent dictionary.
  - `threshold (float, optional)`: Minimum probability for the intent. Defaults to 0.5.
- **Returns:**
  - `str`: Name of the intent or None.

### `get_mentioned_entities(self, intent: dict) -> dict`

Retrieves the entities mentioned in the recognized intent.

- **Parameters:**
  - `intent (dict)`: Recognized intent dictionary.
- **Returns:**
  - `dict`: Dictionary of mentioned entities or None.

### `is_addressed_to_drone(self, intent: dict, name: str = 'sammy', similar_pronunciation: list = None) -> bool`

Checks if the intent is addressed to the drone.

- **Parameters:**
  - `intent (dict)`: Recognized intent dictionary.
  - `name (str, optional)`: Drone's name. Defaults to 'sammy'.
  - `similar_pronunciation (list, optional)`: Names similar to the drone's name.
- **Returns:**
  - `bool`: True if addressed to the drone, otherwise False.

---

---

# `GPTIntentRecognition` 

## Overview

The `GPTIntentRecognition` class integrates with the ChatGPT model for the purpose of recognizing user intents, especially tailored for drone actions.

## Attributes:

- **configs (GPTConfigs):** 
  - Description: Configurations for the GPT Engine.
- **gpt_engine (GPTEngine):** 
  - Description: The GPT engine instance utilized for intent recognition.

## Methods

### `__init__(self, configs: GPTConfigs) -> None`

Initializes the `GPTIntentRecognition` with given configurations and sets up the default system prompt.

- **Parameters:**
  - `configs (GPTConfigs)`: Configurations for the GPT Engine.

### `set_custom_actions_to_system_prompt(self, custom_actions: list) -> None`

Updates the system prompt with custom drone actions.

- **Parameters:**
  - `custom_actions (list)`: List of custom drone actions.

### `get_system_prompt(self) -> str`

Fetches the current system prompt.

- **Returns:**
  - `str`: The current system prompt.

### `override_system_prompt(self, system_prompt: str) -> None`

Replaces the existing system prompt with a new one.

- **Parameters:**
  - `system_prompt (str)`: The desired system prompt.

### `recognize_intent(self, user_message: str) -> str`

Identifies the intent from the supplied user message using the ChatGPT engine.

- **Parameters:**
  - `user_message (str)`: The user's input message for which the intent should be recognized.
- **Returns:**
  - `str`: Recognized intent based on the user's message.

---

---

#  Object Detection

A simple Python module for object detection and bounding box retrieval using various vision algorithms.

## Features

- Supports YOLO V8 for object detection.
- Easily extensible to other algorithms like Google Vision (to be implemented).

## Functions

- `detect_objects(algorithm, vision_config, frame)`: Detects objects in a given frame using the specified vision algorithm.
- `get_bounding_boxes(algorithm, vision_config, frame)`: Retrieves bounding boxes for objects in a given frame using the specified vision algorithm.

## Usage

```python
from vision_object_detection import detect_objects, get_bounding_boxes
from vision_object_detection.enums import VisionAlgorithm
from vision_object_detection.configs import VisionConfigs

config = VisionConfigs(weights_path="path_to_weights_file")
frame = "path_to_image_or_frame_data"

# Detect objects using YOLO V8
detected_objects = detect_objects(VisionAlgorithm.YOLO_V8, config, frame)

# Get bounding boxes using YOLO V8
bounding_boxes = get_bounding_boxes(VisionAlgorithm.YOLO_V8, config, frame)
```

# Vision Engine 

## Overview
This module provides tools for object detection using computer vision techniques. Two main classes are defined: `VisionEngine` and `YoloEngine`.

## Imports
- `cv2`: OpenCV library for computer vision tasks.
- `numpy`: For numerical operations.
- `pkg_resources`: To find and access resources inside packages.
- `get_logger`: A utility function for logging.

## Classes

---

## `VisionEngine`

This class offers a foundational structure designed for object detection tasks.

### Methods

### `init_engine() -> None`

Initializes the vision engine. Intended to be overridden by subclasses for unique initialization steps.

- **Parameters:** None
- **Returns:** None

### `get_object_list(frame: Image) -> List[Object]`

Retrieves a collection of objects detected in the specified frame.

- **Parameters:**
  - `frame`: An image frame where object detection will be carried out.
- **Returns:** 
  - List of detected objects.

### `get_bounding_box(frame: Image) -> List[BoundingBox]`

Generates bounding boxes of the detected objects in the given frame.

- **Parameters:**
  - `frame`: Image frame from which bounding boxes are derived.
- **Returns:** 
  - List of bounding boxes.

---

## `YoloEngine`

This class specializes in utilizing the YOLO (You Only Look Once) object detection technique and is a direct subclass of `VisionEngine`.

### Methods

### `__init__(weights_path: str) -> None`

Initiates the YOLO object detection engine using the specified weights file path.

- **Parameters:**
  - `weights_path`: File path leading to the pre-trained weights.
- **Returns:** None
- **Raises:** 
  - `FileNotFoundError`: If the designated configuration or labels file is not located.

### `init_engine() -> None`

Placeholder method meant for initializing the engine. This method is designed for further elaboration by subsequent implementations or subclasses.

- **Parameters:** None
- **Returns:** None

### `__get_output_layers(net: Network) -> List[str]`

Acquires the names of output layers from the provided network.

- **Parameters:**
  - `net`: Pre-loaded YOLO network.
- **Returns:** 
  - A list of names corresponding to the output layers.

### `get_object_list(image: Image) -> List[str]`

Extracts the labels of objects spotted in the provided image using the YOLO method.

- **Parameters:**
  - `image`: Image in which objects are to be detected.
- **Returns:** 
  - List of labels matching the identified objects.

### `get_bounding_box(image: Image) -> List[BoundingBox]`

Yields the bounding boxes of the detected objects in the provided image by leveraging the YOLO technique.

- **Parameters:**
  - `image`: Image in which objects are to be detected.
- **Returns:** 
  - A list of bounding boxes corresponding to the identified objects.

---
Here's the markdown API documentation for the `SpeechGenerationEngine` class based on the provided code:

---

## `SpeechGenerationEngine`

Handles speech generation based on the provided algorithm and configuration.

### Methods

---

### `__init__(self, algorithm: SpeechGenerationAlgorithm, speech_config: SpeechConfigs)`

Initializes the `SpeechGenerationEngine` with a specific algorithm and configuration.

- **Parameters:**
  - `algorithm (SpeechGenerationAlgorithm)`: Specifies which speech generation algorithm to use.
  - `speech_config (SpeechConfigs)`: Configuration for the speech generation.

- **Returns:** None

---

### `read_aloud(self, phrase: str)`

Converts the provided phrase to speech and plays it aloud.

- **Parameters:**
  - `phrase (str)`: Text that needs to be converted to speech.
  
- **Returns:**
  - The result of the `generate_speech_and_play` method from the initialized speech generation engine (the specific return type is not provided in the given code).

---

### Examples:

```python
engine = SpeechGenerationEngine(SpeechGenerationAlgorithm.GOOGLE_TTS_OFFLINE, config)
engine.read_aloud("Hello, how are you?")
```

---

This markdown content can be integrated into any markdown-supported documentation platform. Adjustments might be needed based on the specific details of your actual implementation or any additional context you might provide.

---

## `OffLineTextToSpeechEngine`

This class serves as a wrapper around the `pyttsx3` library, providing offline text-to-speech capabilities.

### Methods

### `__init__(self, rate=150, volume=1, voice_id='TTS_MS_EN-US_ZIRA_11.0')`

Initializes and configures a text-to-speech engine for speech generation.

- **Parameters:**
  - `rate (int)`: The speech rate in words per minute. Default is 150.
  - `volume (float)`: The speech volume level. Default is 1.0.
  - `voice_id (str)`: The identifier of the desired voice. Default is 'TTS_MS_EN-US_ZIRA_11.0'.

- **Notes:** 
  - Since this is the offline model, it can only support this voice for the moment.

### `generate_speech_and_play(self, text)`

Generates speech from the provided text using a text-to-speech engine and plays it.

- **Parameters:**
  - `text (str)`: The text to be converted into speech and played.

### Examples:

```python
engine = OffLineTextToSpeechEngine()
engine.generate_speech_and_play("Hello, how can I assist you?")
```

---

## `Voice`

This class provides functionality to manipulate and use different voice properties for text-to-speech.

### Methods

### `__init__(self, r, v)`

Initializes a Voice instance with a specified rate and volume.

- **Parameters:**
  - `r`: Rate of speech.
  - `v`: Volume level.

### `get_rate(self) -> int`

Retrieves the current speech rate.

### `get_volume(self) -> float`

Fetches the current speech volume level.

### `set_rate(self, new_rate)`

Updates the speech rate to the given value.

- **Parameters:**
  - `new_rate`: The desired new rate for speech.

### `set_volume(self, new_volume)`

Modifies the volume of the voice.

- **Parameters:**
  - `new_volume`: Desired volume level, should be between 0 and 1.

### `set_voice_id(self, new_voice_id)`

Changes the texture of the voice, such as language and gender.

- **Parameters:**
  - `new_voice_id`: Identifier for the desired voice. For more voice_ids, refer to the documentation.

### `play_audio(self, text)`

Converts a given text into audio and plays it.

- **Parameters:**
  - `text (str)`: The text to be converted to audio and played.

### Examples:

```python
voice = OffLineTextToSpeechEngine.Voice(200, 0.8)
voice.play_audio("Hello, this is a test.")
```

---


---

## `TrackingEngine`

Description of the `TrackingEngine` class.

### Methods

---

### `__init__(self)`

Description of the constructor.

- **Parameters:** None
- **Returns:** None

---

### `init_tracker(self, path: str)`

Initializes a tracker with the necessary files.

- **Parameters:**
  - `path (str)`: The absolute path to the directory containing the two `.pth` files required for the tracker.
  
- **Returns:**
  - `TrackingEngine`: The initialized tracker.

---

### `_build_init_info(self, box)`

Description of the constructor.

- **Parameters:**
  - `box`: Details not specified in the given code.
  
- **Returns:** Not specified in the provided code.

---

### `set_target(self, image_frame: list, box: list)`

Sets the `TrackingEngine` to track a certain object.

- **Parameters:**
  - `image_frame (list)`: The current frame.
  - `box (list)`: The bounding box of the object to be tracked.
  
- **Returns:** None

---

### `get_tracked_bounding_box(self, image_frame: list)`

Locates the object in an image.

- **Parameters:**
  - `image_frame (list)`: The current frame.
  
- **Returns:**
  - `list` or `bool`: Returns the bounding box of the tracked object in this frame. If no target is found, returns `False`.

---

### Examples:

Suppose you have initialized a `TrackingEngine` instance and have a frame and a bounding box:

```python
engine = TrackingEngine()
engine.init_tracker("/path/to/directory/")
engine.set_target(current_frame, bounding_box)
result = engine.get_tracked_bounding_box(another_frame)
```

---


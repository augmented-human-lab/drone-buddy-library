# DroneBuddy Library

# <a name="_lqvy5ze9ihlj"></a> Offline

## <a name="_r98z1rl4ow8h"></a>Atoms

### <a name="_60bsx9mrqz86"></a>basic\_tracking module

#### *class* TrackEngine()

Bases: **object**

##### Function: `_build_init_info(box)`

**Description: The** TrackEngine **class facilitates object tracking in video frames using a specified tracker.**

- __init__(self)
    - **Initializes an instance of the** TrackEngine **class.**
    - **Parameters: None**

##### Function: ` _build_init_info(box)`

**Description: Builds and returns initialization information for a particular bounding box.**

- **Parameters:**
    - **box (Box): The box for which initialization information is needed.**

- **Returns: A dictionary containing initialization information with the following keys:**
    - **init\_bbox** (Box): The initial bounding box.
    - **init\_object\_ids** (list): A list of initial object IDs.
    - **'object\_ids' (list)**: A list of object IDs.
    - **sequence\_object\_ids**(list): A list of object IDs for the sequence.

#### Function: ` init_tracker(path: str)`

**Description**: Initializes a tracker and returns a **TrackEngine** object.

- **Parameters:**
    - **path**(str): The absolute path to the directory containing the required .pth files for the tracker.
- **Returns**:
  - An initialized TrackEngine object configured with the specified tracker.

#### Function: `set_target(img: list, box: list, tracker: TrackEngine)`

**Description**: Sets the TrackEngine to track a specific object in the given image.

- **Parameters:**
    - **img**(list): The current frame (image).
    - **box**(list): The bounding box of the object to be tracked.
    - **tracker**(TrackEngine): The initialized tracker.

#### Function: `get_tracked_bounding_box(img: list, tracker: TrackEngine)`

**Description**: Locates the object in an image using the initialized tracker.

- **Parameters:**
    - **img (list)**: The current frame (image).
    - **tracker (TrackEngine)**: The initialized tracker.
- **Returns**: 
  - The bounding box of the tracked object in the current frame, or False if no target is found.

### <a name="_m6pqb16mbavu"></a>face\_recognition module

#### Function: `add_people_to_memory(name, image_path)`

**Description**: Adds a new person’s name and image to the recognition memory.

- **Parameters:**

  - **name** (*str*) – Name of the new person.
  - **image\_path** (*str*) – Path to the image of the new person.

- **Returns:**

  - True if the addition was successful, False otherwise.

- **Return type:**

  - bool

#### Function: `find_all_the_faces(frame, show_feed=False)`

**Description**:  Detects and recognizes faces in a given video frame.

- **Parameters:**

  - **frame** (*numpy.ndarray*) – The input frame to process.
  - **show_feed** (*bool*) – Whether to display the annotated video feed.

- **Returns:**

  - List of recognized face names.

- **Return type:**

  - list

#### Function: - `get_video_feed**(frame, face_locations, face_names)`

**Description**: Displays the video feed with annotated face boxes and names.

- **Parameters:**

  - **frame** (*numpy.ndarray*) – The frame to display.
  - **face_locations** (*list*) – List of face locations.
  - **face_names** (*list*) – List of recognized face names.

  **Returns:** 
  - None

#### Function: `load_known_face_encodings(known_face_names)`

**Description**: Loads known face encodings corresponding to the provided face names.

- **Parameters:**

  - **known_face_names** (*list*) – List of known face names.

- **Returns:**
  - List of known face encodings.

- **Return type:**
  -   list

#### Function: `load_known_face_names()`

**Description**: Loads a list of known face names from a text file.

- **Returns:**

  - List of known face names.

- **Return type:**

  - list


#### Function: `process_frame_for_recognition(frame)`

**Description**: Preprocesses a video frame for faster face recognition processing.

- **Parameters:**

  - **frame** (*numpy.ndarray*) – The input frame to be processed.

- **Returns:**

  - The processed frame ready for face recognition.

- **Return type:**

  - numpy.ndarray .

#### Function : `read_file_into_list(filename)`

**Description**: Reads lines from a file and returns them as a list.

- **Parameters:**

  - **filename** (*str*) – Path to the file to be read.

- **Returns:**

  - List of lines read from the file.

**Return type:**

- list

**Raises:**

- **FileNotFoundError** – If the specified file is not found.

### <a name="_tuvqe7aq0k2k"></a>gesture\_recognition module

#### Function: `is_pointing(landmarks)`

**Description**: Check whether the hand is in the pointing gesture.

- **Parameters:**

  - **landmarks** (*list*) - the hand landmarks from hand\_detection

- **Returns:**

  - whether the hand is in the pointing gesture

- **Return type:** 
  - bool

#### Function: `is_stop_following(landmarks)`

**Description**:Check whether the hand is in the fisted gesture.

- **Parameters:**

  - **landmarks** (*list*) – the hand landmarks from hand\_detection

- **Returns:**

  - whether the hand is in the fisted gesture

- **Return type:**

  - bool

### hand_control module

#### **Function**: `count_fingers(frame, show_feedback=False)`

**Description**:

The count_fingers function analyzes a video frame containing a hand and counts the number of raised fingers.

- **Parameters**:

  - frame (numpy.ndarray): The input frame containing the hand to be analyzed.
  - show\_feedback (bool, optional): Whether to display the annotated video frame (default is False).

- **Returns**: 
  - An integer representing the count of raised fingers in the hand.

- **Notes**:

  - This function assumes the use of the MediaPipe library for hand tracking.

  - The function also calculates and displays the frames per second (FPS) on the video frame.

**Example**:

- Usage of the function:

```
finger_count = count_fingers(frame, show\_feedback=True)
```

### <a name="_9ww99skpw1j7"></a>hand\_detection module

#### Functions : `get_hand_landmark(img)`

**Description**: Detect hands in an image.

- **Parameters:**

  - **img** (*list*) – the frame to detect the hand in

- **Returns:**

  - return the list of the landmark of one hand in the frame. Return false if no hand is detected.

- **Return type:** 
  - list | bool

### <a name="_47btybqxk8hb"></a>head\_bounding module

#### Function : `get_head_bounding_box(tello)`

**Description**: Get the bounding box of the head in front of the drone.

- **Parameters:**

  - **tello** (*Tello*) –

- **Returns:**

  - image, [int: x coordinate of the left top corner of the bounding box, int: y coordinate of the left top corner of the bounding box, int: width, int: height]

- **Return type:**
  - list

### <a name="_cnvwo8dpmmhs"></a>intent\_recognition module

#### **Function**: `init_intent_recognition_engine(dataset_path: str = None, config: str = CONFIG_EN)`

**Description**:

Initialize the intent recognition engine using the provided dataset file.

- **Parameters**:

  - dataset\_path (str, optional): Path to the JSON dataset file containing the intents and their corresponding
    utterances. If not provided, the default dataset is used.
  - config (str, optional): The configuration to use for the SnipsNLUEngine. Default is English (CONFIG\_EN).

- **Returns**: 
  - An instance of the SnipsNLUEngine that has been trained for intent recognition.

**Example**:

```
engine = init_intent_recognition_engine(dataset_path="path/to/dataset.json", config=CONFIG_EN)
```

#### **Function**: `recognize_intent(engine: SnipsNLUEngine, text: str)`

**Description**:

Given a trained SnipsNLUEngine and a string of text, this function parses the text and returns a dictionary representing
the detected intent and associated slots.

- **Parameters**:

  - **engine** (SnipsNLUEngine): An instance of SnipsNLUEngine, which has been trained on a dataset of intents.
  - **text** (str): A string of text to be parsed by the NLU engine.

- **Returns**: A dictionary representing the detected intent and associated slots, with the following keys:

  - **input**: The given input text.
  - **intent**: An object containing intentName and probability.
  - **slots**: A dictionary containing key-value pairs of detected slots.

**Example**:

```
intent_data = recognize_intent(engine, "Can you turn on the lights in the living room?")
```

#### **Function**: `get_intent_name(intent, threshold=0.5)`

**Description**:

Retrieves the name of the recognized intent based on the provided intent object.

- **Parameters**:

  - **intent** (dict): The intent object containing information about the recognized intent.
  - **threshold** (float, optional): The probability threshold for considering the intent as valid (default is 0.5).

- **Returns**: 
  - A string representing the name of the recognized intent if its probability is above the threshold,
  otherwise None.

**Example**:

``` intent_name = get_intent_name(intent_obj, threshold=0.6) ```

#### **Function**: `get_mentioned_entities(intent)`

**Description**:

Retrieves the key-value pairs from the slots of an intent.

- **Parameters**:

  - **intent** (dict): The intent object containing slots.

- **Returns**: 
  - A dictionary containing the key-value pairs extracted from the slots. Returns None if the intent is None,
  slots are None, or if there are no slots.

**Function**: `is_addressed_to_drone(intent, name='sammy', similar_pronunciation=None)`

**Description**:

Checks if the intent is addressed to the drone.

- **Parameters**:

  - **intent** (dict): The intent object containing slots.
  - **name** (str, optional): The name in which the drone is to be addressed. This should be the same name that the intent
    classifier is trained with (default is 'sammy').
  - **similar\_pronunciation** (list, optional): A list of names that sound similar to the name of the drone.

- **Returns**:

  - True if the intent is addressed to the drone, False otherwise.

**Example**:

```
 addressed_to_drone = is_addressed_to_drone(intent, name='sammy', similar_pronunciation=['samy', 'samie'])
```

### <a name="_o6g46wpyhqz8"></a>object\_detection\_yolo module

#### *class* ` YoloEngine(weights_path)`

Bases: **object**

#### Function:` get_boxes_yolo(yolo_engine, image)`

Get the bounding boxes of objects detected in an image using a YOLO (You Only Look Once) object detection engine.

- **Parameters:**

  - **yolo_engine** () –
    The YoloEngine object used for object detection.

  - **image** – The image to detect objects in.

- **Returns:**

  - A list of bounding boxes corresponding to the objects detected in the image.

#### Function: ` get_label_yolo**(yolo_engine, image)`

Get the labels of objects detected in an image using a YOLO (You Only Look Once) object detection engine.

- **Parameters:**

  - **yolo_engine**  – The YoloEngine object used for object detection.
  - **image** – The image to detect objects in.

- **Returns:**

  - A list of labels corresponding to the objects detected in the image.


#### Function: `init_yolo_engine(weights_path)`

Initialize a YOLO (You Only Look Once) object detection engine.

- **Parameters:**

  - **weights\_path** (**str**) – The file path to the pre-trained weights file.

- **Returns:**

  - None.

- **Raises:**

  - **FileNotFoundError** – If the specified configuration or labels file is not found.

### <a name="_uj414tbjwnnu"></a>object\_memorize module

#### **Function**: ` update_memory() `

**Description**:

Update the memory by retraining the model. This process may take several minutes to complete.

- **Parameters**: 
  - None

- **Returns:** 
  - None

### <a name="_ky18hcbbhmof"></a>object\_selection module


#### `select_pointed_obj(frame, landmarks, obj_result)`

**Description**:

Pick out the object the finger is pointing to.

- **Parameters:**

  - **frame** (*list*) – the current frame
  - **landmarks** (*list*) – the landmarks detected by hand\_detection in this frame
  - **obj\_result** (*list*) – the bounding boxes of the objects detected in this frame

- **Returns:**

  - the bounding box of the pointed object

- **Return type:**

  - list


### <a name="_uhfv2im32rj0"></a>speech\_2\_text\_conversion module

#### `init_speech_to_text_engine(language)`

**Description**:

Initializes a speech-to-text engine using the Vosk model for a given language. (currently only supports ‘en-US’
language)

- **Parameters**:

  - language: a string representing the language code to use (e.g. ‘en-us’, ‘fr-fr’)

- **Returns**:

  - a Vosk KaldiRecognizer object that can be used for speech recognition

- **Notes**:

`
The language of the model. The default is ‘en-US’. (currently only supports this language) :
return: The vosk model.
need to initialize the model before using the speech to text engine.`

- **Return Type:**

  - param language

#### `recognize_command(model, audio_feed)`

**Description**:

Recognizes a command from an audio feed using a given model.

- **Parameters**:

  - model: The vosk model that is returned by the init_speech_to_text_engine().
  - audio\_feed: a byte string representing the audio feed to recognize, taken by audio_feed.read(num_frames)

- **Returns**:

  - a label indicating the recognized command, or None if no command was recognized

#### `recognize_speech(model, audio_feed)`

**Description**:

Recognizes a text from an audio feed using a given model.

- **parameters**:

  - model: The vosk model that is returned by the init\_speech\_to\_text\_engine().
  - audio\_feed: a byte string representing the audio feed to recognize, taken by audio_feed.read(num_frames)

**Returns**:
  - the text that was recognized, or None if no text was recognized

### <a name="_f3mw07q72bic"></a>text\_2\_speech\_conversion module

#### *class* `Voice(r, v)`

Bases: **object**
  
  **get\_rate**()
  
  **get\_volume**()
  
  **play_audio**(*text*)
  
  **set\_rate**(*new\_rate*)
  
  **set_voice_id**(*new_voice_id*)
  
  **set_volume**(*new_volume*)

##### Function: `generate_speech_and_play(engine, text)`

**Description**:

Generates speech from the provided text using a text-to-speech engine and plays it.

- **Parameters:**

  - **engine** (*TTS Engine*) – The text-to-speech engine instance capable of generating speech.
  - **text** (*str*) – The text to be converted into speech and played.

- **Returns:**

  - None

**Example**

```
engine = TextToSpeechEngine() generate_speech_and_play(engine, “Hello, how can I assist you?”)
```

#### FUnction: `init_text_to_speech_engine(rate=150, volume=1, voice_id='TTS_MS_EN-US_ZIRA_11.0')`
**Description**:

Initializes and configures a text-to-speech engine for generating speech.

- **Parameters**:

  - rate (int): The speech rate in words per minute (default is 150).
  - volume (float): The speech volume level (default is 1.0).
  - voice\_id (str): The identifier of the desired voice (default is ‘TTS_MS_EN-US_ZIRA_11.0’).

- **Returns**:

  - pyttsx3.Engine: The initialized text-to-speech engine instance.

**Example**:

```
engine = init_text_to_speech_engine(rate=200, volume=0.8, voice_id=’TTS_MS_EN-US_DAVID_11.0’) 
generate_speech_and_play(engine, “Hello, how can I assist you?”)
```

## <a name="_1cve3b2wwij1"></a>**Molecules**

### <a name="_ukhuap75vqs"></a>fly\_around module

#### **Class**: FlyArrounder

**Description**:

A class representing a drone's flight behavior for capturing images of memorized objects.

- **Constructor**: `__init__(self, tello: Tello, name: str, tracker: Tracker)`

  -   Initializes the FlyArrounder instance.

- **Parameters**:

  - tello (Tello): The Tello drone instance.
  - name (str): The name of the memorized object.
  - tracker (Tracker): The object tracker instance for tracking the memorized object.

#### **Function**: `cut(self, img_name: str)`

**Description**:

Captures an image of the tracked memorized object from the drone's camera feed.

- **Parameters**:

  - img_name (str): The name to be given to the captured image.
  
#### Function: `init_fly_arrounder(tello: Tello, name: str, tracker: Tracker)`

**Description**:

Initiates an engine for flying around the object.

- **Parameters**:

  - tello (Tello): The Tello drone instance.
  - name (str): The label for the object.
  - tracker (Tracker): An initialized tracker.

- **Returns**:

  - FlyArrounder: The initialized FlyArrounder engine.

#### **Function**: `fly_around(fly_arrounder: FlyArrounder, frame, box)`

**Description**:

Fly around and take photos of the object for memorization.

- **Parameters**:

  - fly_arrounder (FlyArrounder): The initialized FlyArrounder engine.
  - frame (list): The current frame.
  - box (list): The bounding box around the object in this frame.

### <a name="_hjkgst97qjur"></a>follow\_me module

#### **Function**: `follow_me(tello: Tello, path: str)`

**Description**:

Follow the person in front of the drone.

- **Parameters**:

  - tello (Tello): The Tello drone instance.
  - path (str): The absolute path to the directory of the two .pth files for the tracker.

### <a name="_qhlxkctvpi52"></a>follower\_engine module

**Class**: Follower

**Description**:

A class representing a drone follower behavior using hand gestures.

**Constructor**: `__init__(self, tracker: TrackEngine, tello: Tello)`

Initializes the Follower instance.

- **Parameters**:

  - tracker (TrackEngine): The object tracker instance for tracking an object.
  - tello (Tello): The Tello drone instance.

#### **Function**: `detect_stop_gesture(self)`

**Description**:

Monitors the camera feed for a stop gesture and terminates the follower behavior if detected.

#### **Function**: `init_follower(tracker: TrackEngine, tello: Tello)`

**Description**:

Initialize a follower.

- **Parameters**:

  - tracker (TrackEngine): An initialized follower with a target set.
  - tello (Tello): The Tello drone instance.

- **Returns**: 
  - Follower: An initialized follower.

#### **Function**: `follow(follower: Follower)`

**Description**:

Control the drone's following behavior.

- **Parameters**:

  - follower (Follower): Initialized follower.

### <a name="_pxrxj1p7fzs7"></a>hand\_follower\_engine module

**Class**: HandFollower

**Description**:

A class representing a drone follower behavior using hand gestures.

**Constructor**: `__init__(self, tello: Tello)`

**Description**: 

Initializes the HandFollower instance.

- **Parameters**:

  - tello (Tello): The Tello drone instance.

#### **Function**: `detect_stop_gesture(self)`

**Description**:

Monitors the camera feed for a stop gesture and terminates the follower behavior if detected.

#### **Function**: init_hand_follower(tello: Tello)

**Description**:

Initialize a hand follower.

- **Parameters**:

  - tello (Tello): The Tello drone instance.

- **Returns**:

  - HandFollower: The initialized hand follower.

#### **Function**: `close_to_hand(follower: HandFollower)`

**Description**:

Get the drone close to the detected hand.

- **Parameters**:

  - follower (HandFollower): Initialized hand follower.

### <a name="_shikbw199t8t"></a>object\_pointer\_engine module

**Function**: `get_pointed_obj(follower: HandFollower, yolo_engine: YoloEngine)`

**Description**:

Get the bounding box of the pointed object and the frame once the drone detects a pointing hand gesture.

- **Parameters**:

  - follower (HandFollower): An initialized hand follower.
  - yolo\_engine (YoloEngine): An initialized YoloEngine for object detection.

- **Returns**:

  - tuple[list, list] | None: A tuple containing the frame including the pointing gesture and the bounding box of the
    pointed object if detected; otherwise, returns None

##

# <a name="_7tci3iswbx38"></a><a name="_mtzd33sm3q8i"></a>Online

## <a name="_qfufp4b0919w"></a>**Atoms**

### <a name="_bs2wp78z25av"></a>online\_conversation\_generation module

**Function**: `prompt_chatgpt(prompt: str)`

**Description**:

Generates a response using the OpenAI GPT model based on the provided prompt.

- **Parameters**:

  - prompt (str): The input prompt to initiate the conversation.

- **Returns**:

  - str: The generated response from the OpenAI GPT model.

### <a name="_rzhgr1lkg11c"></a>online\_speech\_2\_text\_conversion module

**Function**: `init_google_speech_engine()`

**Description**:

Initializes the Google Cloud Speech-to-Text client.

- **Returns**:

  - speech.SpeechClient: The initialized Speech-to-Text client.
- **Notes**:


-   This function initializes the Google Cloud Speech-to-Text client and returns an instance of the client that can be used
to recognize speech from audio streams.

#### **Function**: ` recognize_speech(client: speech.SpeechClient, audio_stream: bytes) -> speech.RecognizeResponse`

**Description**:

Recognizes speech from an audio stream using the Google Cloud Speech-to-Text client.

- **Parameters**:

  - client (speech.SpeechClient): The Speech-to-Text client instance.
  - audio\_stream (bytes): The audio stream content to be recognized.

- **Returns**:

  - speech.RecognizeResponse: The response containing recognized speech results.

### <a name="_qa8mkhfaoe7f"></a>online\_text\_recognition module

#### **Function**: ` init_google_vision_engine() `

**Description**:

Initializes the Google Cloud Vision client for image annotation.

- **Returns**: 
  - vision.ImageAnnotatorClient: The initialized Vision client.

**Notes**:

This function initializes the Google Cloud Vision client for image annotation and returns an instance of the client that
can be used to perform various image analysis tasks.

#### **Function**: ` detect_text(client: vision.ImageAnnotatorClient, image_path: str) -> google.protobuf.json_format.MessageToJson `

**Description**:

Detects text in an image using the Google Cloud Vision client.

- **Parameters**:

  - client (vision.ImageAnnotatorClient): The Vision client instance.
  - image\_path (str): The path to the image file to be analyzed.

- **Returns**: 
  - google.protobuf.json\_format.MessageToJson: The response containing detected text annotations in JSON
  format.


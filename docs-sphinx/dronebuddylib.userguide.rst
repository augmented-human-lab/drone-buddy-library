User Guide
==========

Introduction
------------


DroneBuddy envisions empowering everyone with the ability to personally program their intelligent drones, enriching them with desired features. At its core, DroneBuddy offers a suite of fundamental building blocks, enabling users to seamlessly integrate these elements to bring their drone to flight.

Functioning as an intuitive interface, DroneBuddy simplifies complex algorithms, stripping away the intricacies to offer straightforward input-output modalities. This approach ensures that users can accomplish their objectives efficiently, without getting bogged down in technical complexities. With DroneBuddy, the focus is on user-friendliness and ease of use, making drone programming accessible and hassle-free.


Voice Recognition
-----------------

General
~~~~~~~

Voice recognition, also known as speech recognition, is a technology that allows computers or machines to understand and interpret spoken language. It enables the conversion of spoken words into written text or commands that can be understood and processed by a computer.

Here's a simplified explanation of how voice recognition works:

#. Audio Input: Voice recognition systems take audio input as their primary source of information. This audio input can be obtained from various sources, such as a microphone, recorded audio files, or even real-time streaming audio. In our case for us to command the drone we will take the input from our computer. There is one more possibility of using the stream from the drone itself, but this will be very messy, there will be a lot of background noise which would not do well with the accuracy. And also this will put too much strain on the drone, which will cause the drone to unnecessarily heat up and drain the battery.

#. Pre-processing: Before analyzing the audio, voice recognition systems often apply pre-processing techniques to enhance the quality of the input. This may involve removing background noise, normalizing the audio volume, or applying filters to improve the accuracy of recognition..

#. Acoustic Analysis: The audio input is then analyzed to extract various acoustic features. These features capture information about the sound, such as the frequency, intensity, and duration of different speech sounds.

#. Acoustic Model: An acoustic model is a trained statistical model that associates the observed acoustic features with the corresponding speech sounds or phonemes. It helps identify and differentiate between different speech sounds in the audio.

#. Language Model: A language model helps in understanding the context and improving the accuracy of recognition. It uses probabilistic methods to predict the most likely sequence of words or phrases based on the acoustic input. Language models incorporate grammar rules, vocabulary, and statistical language patterns to generate the most probable textual output.

#. Speech Recognition: In this step, the acoustic model and the language model work together. The observed acoustic features are compared with the models to determine the most likely word sequence that corresponds to the input speech. This involves matching the acoustic patterns against a vast database of pre-recorded speech samples.

#. Output Generation: The recognized words or phrases are generated as output, typically in the form of written text. This output can be further processed or used as input for various applications, such as transcription services, voice assistants, or voice-controlled systems.

It's important to note that voice recognition technology is continually evolving and improving. However, it can still face challenges in accurately recognizing speech due to factors like background noise, accents, speech impediments, and variations in pronunciation.

Voice recognition has numerous applications, including dictation software, transcription services, virtual assistants, interactive voice response systems, and more. It offers a convenient and efficient way for humans to interact with computers and devices through spoken language.


.. toctree::
   :maxdepth: 3

   dronebuddylib.userguide.voicerecognition

Intent Recognition
-----------------

General
~~~~~~~

Intent recognition, also known as intent detection or intent classification, is a technique used in natural language processing (NLP) to identify the intention or purpose behind a given piece of text. It helps computers understand the meaning and intention behind human language, enabling them to respond appropriately.

Here's a simplified explanation of how intent recognition works:

#. Text Input: Intent recognition takes a text input, typically a user's query, command, or statement, as its primary source of information. This text input can be in the form of a sentence, a phrase, or even a single word.

#. Pre-processing: Before analyzing the text, intent recognition systems often perform pre-processing steps. These steps may include removing punctuation, converting the text to lowercase, removing stop words (common words like "the," "is," etc.), and handling any other necessary formatting or cleaning.

#. Feature Extraction: Intent recognition systems extract relevant features from the pre-processed text. These features can include words, word combinations (n-grams), part-of-speech tags, syntactic structures, or any other linguistic information that helps capture the meaning and context of the text.

#. Training Data: Intent recognition models require training data to learn how to classify different intents. Training data consists of labeled examples where each text input is associated with a specific intent. For instance, if the system is designed to recognize user commands, the training data might contain examples like "Play a song," "Stop the video," etc., along with their corresponding intents.

#. Machine Learning Model: Machine learning techniques, such as supervised learning, are commonly used for intent recognition. A model is trained on the labeled training data, learning the patterns and relationships between the extracted features and the corresponding intents.

#. Intent Classification: Once the model is trained, it can be used to predict the intent of new, unseen text inputs. The trained model takes the extracted features from the new text input and applies the learned patterns to classify it into one or more predefined intents. The predicted intent represents the underlying meaning or purpose of the text.

#. Output Generation: The predicted intent is generated as output, providing information about the user's intention. This output can be further processed to trigger specific actions, retrieve relevant information, or provide appropriate responses based on the recognized intent.

Intent recognition is widely used in various applications, such as chatbots, virtual assistants, customer support systems, and voice-controlled devices. It enables these systems to understand user queries, commands, or statements and respond accordingly, providing a more interactive and personalized user experience.

.. important::
    It's important to note that the accuracy of intent recognition depends on the quality and diversity of the training data, the effectiveness of feature extraction techniques, and the robustness of the machine learning model employed.


.. toctree::
   :maxdepth: 3

   dronebuddylib.userguide.intentrecognition

Face Recognition
-----------------

General
~~~~~~~

Face recognition is a technology that identifies or verifies a person's identity by analyzing their facial features. It is commonly used in various applications, such as security systems, access control, biometric authentication, and surveillance.

Here's a simplified explanation of how face recognition works:

#. Face Detection: The face recognition process begins with face detection, where an algorithm locates and detects faces within an image or a video stream. This involves analyzing the visual information to identify areas that potentially contain faces.

#. Face Alignment: Once faces are detected, face alignment techniques are applied to normalize the face's position and orientation. This step helps ensure that the subsequent analysis focuses on the important facial features, regardless of slight variations in pose or facial expression.

#. Feature Extraction: In this step, the facial features are extracted from the aligned face image. Various algorithms, such as eigenfaces, local binary patterns, or deep neural networks, are used to analyze the unique characteristics of the face and represent them as numerical feature vectors. These feature vectors capture information about the geometry, texture, and other distinctive aspects of the face.

#. Training: Face recognition systems require training with labeled examples to learn how to identify individuals. During the training phase, the system learns to map the extracted facial features to specific identities. It builds a model or database that represents the known faces and their corresponding features.

#. Face Matching: When a face needs to be recognized, the system compares the extracted features of the query face with the features stored in the trained model or database. It calculates the similarity or distance between the feature vectors to determine the closest matches. The recognition algorithm uses statistical methods or machine learning techniques to make this comparison.

#. Recognition Decision: Based on the calculated similarity scores, the system makes a recognition decision. If a sufficiently close match is found in the database, the face is recognized as belonging to a specific individual. Otherwise, if the match is not close enough, the system may classify the face as unknown.

Face recognition systems can be designed for different purposes, such as one-to-one verification (confirming whether a person is who they claim to be) or one-to-many identification (finding a person's identity from a large database). The level of accuracy and performance can vary based on factors such as image quality, variations in lighting and pose, and the quality of the face recognition algorithm used.


.. toctree::
   :maxdepth: 3

   dronebuddylib.userguide.facerecognition

Object detection
-----------------

General
~~~~~~~

Object detection is a computer vision technique that involves locating and classifying objects within images or video frames. It goes beyond simple image classification by not only identifying the presence of objects but also providing information about their precise locations within the scene.

Here's a simplified explanation of how object detection works:

#. Image Preprocessing: The object detection process typically starts with some preprocessing steps, such as resizing, normalization, or adjusting the image's color channels. These steps help prepare the image for further analysis and improve the accuracy of object detection algorithms.

#. Feature Extraction: Object detection algorithms employ various techniques to extract features from the image that represent meaningful patterns or characteristics of objects. Commonly used methods include the Histogram of Oriented Gradients (HOG), Scale-Invariant Feature Transform (SIFT), or Convolutional Neural Networks (CNNs). These features capture essential information about the objects' appearance, shape, and texture.

#. Object Localization: Object localization involves determining the spatial coordinates of objects within the image. This is typically achieved by identifying the boundaries of objects through techniques like edge detection, contour detection, or region proposal algorithms. The result is a bounding box that tightly encloses each detected object.

#. Classification: Once objects are localized, object detection algorithms assign class labels to each detected object. Classification can be performed using machine learning techniques like Support Vector Machines (SVMs), Random Forests, or CNNs. The model has been previously trained on a dataset with annotated examples of different object classes, allowing it to learn to recognize and classify objects accurately.

#. Post-processing: In the post-processing step, the object detection algorithm refines the results to improve the overall quality. This may involve filtering out detections based on their confidence scores, removing overlapping or redundant bounding boxes, or applying heuristics to handle edge cases or false positives.

Object detection can be applied to a wide range of applications, including autonomous driving, surveillance systems, robotics, and image understanding tasks. The accuracy and performance of object detection algorithms depend on factors such as the quality of training data, the choice of features and algorithms, and the computational resources available.

It's important to note that object detection is an active area of research, and there are several algorithms and frameworks available to perform object detection, including Faster R-CNN, YOLO (You Only Look Once), and SSD (Single Shot MultiBox Detector). These algorithms differ in their trade-offs between accuracy and speed, making them suitable for different use cases.


.. toctree::
   :maxdepth: 3

   dronebuddylib.userguide.objectdetection


Voice generation
-----------------

General
~~~~~~~

Text-to-speech (TTS) is a technology that converts written text into spoken words. It enables computers and other devices to generate human-like speech by processing and synthesizing text input.

Here's a simplified explanation of how text-to-speech works:

Text Processing: The input text is processed to remove any unwanted characters, punctuation, or formatting. It may also involve tokenization, which breaks the text into smaller units such as words or phonemes for further analysis.

Linguistic Analysis: The processed text is analyzed to extract linguistic features and interpret the meaning of the words and sentences. This analysis may involve tasks like part-of-speech tagging, syntactic parsing, and semantic understanding to ensure accurate pronunciation and intonation.

Speech Synthesis: Once the linguistic analysis is complete, the text is transformed into speech signals. This is typically done through speech synthesis algorithms that generate the corresponding waveforms based on the linguistic information.

Concatenative Synthesis: One approach is concatenative synthesis, where pre-recorded segments of human speech (called "units") are stored in a database. These units are selected and concatenated together to form the synthesized speech. This method can produce natural-sounding results but requires a large database of recorded speech.

Formant Synthesis: Another approach is formant synthesis, which generates speech by modeling the vocal tract and manipulating its resonances (formants). By controlling the formants' frequencies and amplitudes, synthetic speech is produced. Formant synthesis allows for more control over the speech output but may sound less natural compared to concatenative synthesis.

Parametric Synthesis: Parametric synthesis uses mathematical models to represent speech characteristics. It employs a set of parameters, such as pitch, duration, and spectral envelope, to generate speech waveforms. Parametric synthesis allows for efficient storage and customization of speech but may require additional processing to sound natural.

Voice and Prosody: TTS systems often have multiple voices available, each representing a different speaker or style. The selected voice determines the characteristics of the generated speech, such as pitch, intonation, and accent. Prosody refers to the rhythm, stress, and intonation patterns in speech, and TTS systems incorporate prosodic rules to make the synthesized speech sound more natural and expressive.

Output: The final output of the TTS system is the synthesized speech, which can be played back through speakers, headphones, or integrated into applications, devices, or services for various purposes like accessibility, voice assistants, audiobooks, or interactive systems.

TTS technology has made significant advancements in recent years, leveraging machine learning and deep neural networks to improve the naturalness and quality of synthesized speech. Deep learning models, such as WaveNet and Tacotron, have demonstrated impressive results in generating high-fidelity and expressive speech.

It's important to note that TTS systems require training on large datasets of recorded speech to produce accurate and natural-sounding results. The quality and performance of TTS systems can vary depending on the available resources, language, voice dataset, and synthesis techniques used.


.. toctree::
   :maxdepth: 3

   dronebuddylib.userguide.voicegeneration


Place Recognition
-----------------

General
~~~~~~~

Place recognition is a computer vision technique that involves identifying and classifying specific locations or landmarks within images or video frames. This technique is essential for applications such as autonomous navigation, robotics, augmented reality, and geographic information systems. Unlike object detection, which focuses on identifying various objects, place recognition specifically targets the identification of locations based on visual cues.

Here’s a simplified explanation of how place recognition works:

1. **Image Preprocessing**: The process typically starts with preprocessing steps such as resizing, normalization, and color channel adjustment. These steps help prepare the image for further analysis and improve the accuracy of the recognition algorithms.

2. **Feature Extraction**: Place recognition algorithms use various techniques to extract features that capture the essential characteristics of the image. Commonly used methods include Convolutional Neural Networks (CNNs) like ResNet, DenseNet, and GoogLeNet. These features encode important information about the appearance, shape, and texture of the scene.

3. **Feature Matching**: Once features are extracted, the algorithm compares them with a database of known features from previously seen locations. This step involves finding the best matches between the features of the input image and those in the database.

4. **Localization and Classification**: After matching features, the algorithm determines the location depicted in the image. This involves identifying the best match from the database and classifying the image accordingly. The classification can be enhanced using techniques like Random Forest classifiers, which have been trained on labeled datasets.

5. **Post-processing**: In this step, the algorithm refines the results to improve overall accuracy. This may involve filtering based on confidence scores, handling false positives, and ensuring robust recognition even under challenging conditions.



It’s important to note that place recognition is an evolving field, with ongoing research improving the accuracy and efficiency of algorithms. Techniques such as the use of deep learning models and advanced feature extraction methods have significantly enhanced the capabilities of place recognition systems.

.. toctree::
   :maxdepth: 3

   dronebuddylib.userguide.placerecognition

Object Identification
---------------------

General
~~~~~~~

Object identification is a computer vision technique that involves locating and classifying objects within images or video frames. This technology is essential for applications such as robotics, surveillance, image retrieval, and autonomous systems. Object identification not only recognizes the presence of objects but also provides detailed descriptions of each identified object.

Here's a simplified explanation of how object identification works:

Image Preprocessing: The process typically starts with preprocessing steps such as resizing, normalization, and color adjustment. These steps help prepare the image for further analysis and improve the accuracy of the identification algorithms.

Feature Extraction: Object identification algorithms use various techniques to extract features that capture the essential characteristics of the image. Commonly used methods include Convolutional Neural Networks (CNNs) like ResNet. These features encode important information about the appearance, shape, and texture of the objects.

Object Identification: The extracted features are sent to a language model like GPT-4, which interprets these features to identify and describe the objects within the image. GPT-4 provides detailed descriptions based on the visual features extracted by the CNN.

Concatenative Recognition: One approach is concatenative recognition, where pre-trained segments of object features are stored in a database. These segments are selected and matched to the features in the input image to form the final identification. This method can produce accurate results but requires a large database of pre-trained object features.

Formant Recognition: Another approach is formant recognition, which identifies objects by modeling their features and manipulating these models to match the input image. This approach allows for more control over the identification process but may require more computational resources.

Parametric Recognition: Parametric recognition uses mathematical models to represent object characteristics. It employs a set of parameters, such as shape, texture, and color, to generate feature vectors. Parametric recognition allows for efficient storage and customization of object features but may require additional processing to ensure accurate identification.

Post-processing: In this step, the algorithm refines the results to improve overall accuracy. This may involve filtering based on confidence scores, handling false positives, and ensuring robust identification even under challenging conditions. Techniques such as non-maximum suppression can be applied to remove duplicate detections and improve the final output.

Output: The final output of the object identification system is the recognized objects, which can be used for various purposes such as guiding robots, monitoring security cameras, or searching for images. The output typically includes the object names, their locations within the image, and the confidence scores of the identifications.

.. toctree::
   :maxdepth: 3

   dronebuddylib.userguide.objectidentification

LLM Integration
---------------

General
~~~~~~~

The LLM integration is a set of scripts that allow to run specific functions with the LLM. The implementation follows multi agent architecture, where each agent is capable of doing a specific task.
Currently, the supported LLM is OpenAI ChatGPT. But the architecture is designed to be easily extendable to other LLMs, which can be hosted locally or on the cloud.
Each agent has a predefined output, specific to the output expected from the LLM.

The LLM integration is a comprehensive set of scripts designed to execute specific functions with a Large Language Model (LLM). The implementation employs a multi-agent architecture, where each agent is dedicated to performing a distinct task. This modular approach enhances the system's efficiency and scalability.

Currently, the supported LLM is OpenAI's ChatGPT. However, the architecture is built with flexibility in mind, allowing for easy extension to support other LLMs. These additional LLMs can be hosted either locally or on the cloud, providing adaptability to various deployment environments.

Each agent within this system has a predefined output, tailored to the expected responses from the LLM, ensuring consistency and reliability in the results.

Understanding LLMs
~~~~~~~~~~~~~~~~~~~

How LLMs Work
~~~~~~~~~~~~~
Large Language Models (LLMs) are advanced artificial intelligence systems trained on vast amounts of text data. They use machine learning techniques to understand and generate human-like text based on the input they receive. These models, such as OpenAI's GPT-3 and GPT-4, leverage deep learning algorithms, particularly transformers, to process and generate language.

System Prompts
~~~~~~~~~~~~~~

System prompts are initial instructions provided to the LLM to guide its behavior and responses. These prompts can set the context, define the role of the AI, or instruct it to follow specific guidelines while interacting. For example, a system prompt can instruct the LLM to act as a customer service representative, ensuring its responses are aligned with customer service protocols.

Session Management
~~~~~~~~~~~~~~~~~~

Sessions with an LLM involve maintaining context over a series of interactions. This is crucial for tasks that require continuity and coherence in conversation. Session management typically includes:

    -   State Tracking: Keeping track of the conversation state to ensure context-aware responses.
    -   Context Preservation: Using tokens or identifiers to maintain session information across different interactions.
    -   Timeouts and Endpoints: Defining session timeouts and endpoints to manage resources effectively and ensure the system does not retain unnecessary data beyond its useful period.

By understanding these components, users can effectively utilize LLMs for a wide range of applications, ensuring robust and coherent interactions.


.. toctree::
   :maxdepth: 3

   dronebuddylib.userguide.llmintegration


Navigation
----------

General
~~~~~~~

Waypoint navigation is a robotics technique that enables autonomous movement between predefined locations or waypoints. Unlike GPS-based navigation systems, waypoint navigation creates detailed movement maps through manual recording and enables precise autonomous navigation in indoor environments where GPS signals are unavailable.

Here's a simplified explanation of how waypoint navigation works:

#. **Manual Mapping Phase**: The navigation process begins with a manual mapping phase where a human operator controls the vehicle (in this case, a drone) through the desired environment. During this phase, the system records all movements including distance, direction, orientation changes, and timing data with high precision.

#. **Movement Recording and Telemetry**: Each movement is tracked using the vehicle's onboard sensors and telemetry systems. The system captures positional data, orientation angles, altitude changes, and movement durations. Movement distances are calculated based on speed, duration, and acceleration/deceleration compensation to ensure accuracy.

#. **Waypoint Creation and Graph Building**: At strategic locations during manual flight, operators can mark their current position as a waypoint. Each waypoint stores the complete sequence of movements required to reach it from the previous waypoint, creating a connected graph of navigable positions throughout the environment.

#. **Data Persistence and Storage**: All waypoint data is stored in structured formats (typically JSON) containing movement sequences, waypoint metadata, timing information, and environmental conditions. This allows for persistent navigation maps that can be reused across multiple navigation sessions.

#. **Pathfinding Algorithms**: The navigation engine calculates paths between any two waypoints using pathfinding algorithms. These include forward navigation (following recorded movements) and reverse navigation (automatically calculating inverse movements by reversing directions and adjusting orientations).

#. **Autonomous Execution**: During autonomous navigation, the system executes the calculated movement sequence with precise timing and orientation control. Each movement is performed with the vehicle oriented to the correct heading before executing linear, rotational, or vertical movements.

#. **Safety and Monitoring Integration**: The system includes comprehensive safety features including real-time monitoring of vehicle status (battery, sensors, connectivity), emergency shutdown capabilities, and graceful error handling to ensure safe operation throughout the navigation process.

Waypoint navigation's key advantages lie in its precision and adaptability. By recording actual movement sequences rather than relying on external positioning systems, it maintains centimeter-level accuracy in challenging environments. The bidirectional pathfinding enables efficient navigation between any waypoints regardless of the original recording sequence.

The algorithm supports multiple operational modes including real-time mapping, interactive waypoint selection, programmatic navigation to specific locations, sequential multi-waypoint missions, and environmental scanning operations. The system is designed for robustness with cross-platform compatibility and platform-specific optimizations.

The navigation system requires initial training through manual exploration sessions where operators map their desired operational areas. The quality and coverage of the resulting navigation map depend on the thoroughness of the initial mapping session and the strategic placement of waypoints throughout the environment.

Keep in mind that while waypoint navigation offers precise autonomous navigation capabilities, it requires an initial mapping phase and is optimized for relatively stable environments. The choice of navigation algorithm depends on the specific requirements of the application, balancing factors like precision, environmental constraints, and available vehicle capabilities.

.. toctree::
   :maxdepth: 3

   dronebuddylib.userguide.navigation

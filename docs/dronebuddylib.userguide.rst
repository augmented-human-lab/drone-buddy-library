User Guide
==========

Introduction
------------

The vision of DroneBuddy is to give the possibility to everyone to programme their intelligent drone by themselves adding all the features they wish. DroneBuddy provides all the basic building blocks so you can use them to make your drone fly.
Best thing about DroneBuddy is that it is an offline library, so you don’t need an active internet connection to carry on the tasks.

Installation
------------
DroneBuddy is just another python library that you can install. You can find the library at https://pypi.org/project/dronebuddylib/

But before the installation DroneBuddy needs some prerequisites.

It is compulsory that you follow all the steps, so that they are installed correctly. These requirements are requested by some of the machine learning models that we use in DroneBuddy.

#.  Python 3.10 or higher
#.  Install visual C++ through https://visualstudio.microsoft.com/visual-cpp-build-tools/, when you are installing there is an option for CMake, select that.
#.  Need to install rust, use https://www.rust-lang.org/tools/install. Follow all the instructions are here
#.  Install CMake and add it to the system path as here
#.  pip install setuptools-rust

Installing Pre-requisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 2

   installing_visual_cpp
   installing_cmake
   installing_rust

Installing visual C++
---------------------

Windows guide
~~~~~~~~~~~~~
#.  Go to this link https://visualstudio.microsoft.com/visual-cpp-build-tools/.
#.  Download Microsoft C++ Build Tools
#.  Install it from the setup, once it is installed, you will get a pop up window, the is an option to install C++
#.  Select Desktop development with C++

Install this option.

.. note::
    Note: This C++ version is required to install CMake which is required to install the face recognition model used to recognize the people that you know.

Install CMake
~~~~~~~~~~~~~~~~~~
CMake is required to install the dlib library which is required to install the face-recognition model. The official guidelines to install can be found here. https://github.com/ageitgey/face_recognition/issues/175#issue-257710508

#. The windows installer can be found here https://cmake.org/download/
#. Once you run it, make sure it is added to the system path. The path would look like C:\Program Files\CMake\bin

Rust installation
~~~~~~~~~~~~~~~~~~

Rust is required to install the intent recognition library that is needed to understand the commands given to the drone.

You can download Rust from here  https://www.rust-lang.org/tools/install

The steps will differ from Windows to mac, so follow the steps on the website.

Windows installation is pretty straightforward

mac OS installation
~~~~~~~~~~~~~


If you get a permission error while installing , follow the https://stackoverflow.com/questions/45899815/could-not-write-to-bash-profile-when-installing-rust-on-macos-sierra steps given in the answer here.


Give a try using this not using sudo:
.. code-block:: bash
    curl https://sh.rustup.rs -sSf | sh -s -- --help
If that works then probably you could try:
.. code-block:: bash
    curl https://sh.rustup.rs -sSf | sh -s -- --no-modify-path

If the command with the --no-modify-path option works, you'll have to manually update .bash_profile to include it in your path:
.. code-block:: bash
    source ~/.cargo/env

Now you are all ready to install DroneBuddy.

Installing DroneBuddy
------------
Installation is as any other library..

.. code-block:: bash

    pip install dronebuddylib

Now let's get into detail regarding all the functionalities DroneBuddy offers.


Once dronebuddylib is installed,

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


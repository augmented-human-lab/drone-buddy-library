Supported models
==========

Face-recognition
~~~~~~~~~~~~~~~~~~~~~~~

Face-recognition is an open-source Python library that provides face detection, face alignment, and face recognition capabilities.

Here's a simplified explanation of how the "face_recognition" library works:

#. Installation: To use the "face_recognition" library, you need to install it first. You can install it via pip by running the following command in your Python environment:

.. code-block:: bash

    pip install face_recognition

#. Face Detection: The library utilizes pre-trained models to detect faces in images or video frames. It can detect multiple faces in an image and return the bounding box coordinates (top, right, bottom, left) for each detected face. The detection process uses computer vision algorithms to locate the presence and location of faces.

#. Face Alignment: After face detection, the library can perform face alignment to normalize the face's position and orientation. This helps improve the accuracy of subsequent face recognition tasks by aligning the faces to a standardized pose.

#. Feature Extraction: The library extracts facial features from the aligned face images. It employs a deep learning-based method to capture important characteristics of the face. These features are represented as numerical feature vectors that encode information about the face's geometry, texture, and other discriminative details.

#. Face Recognition: Using the extracted feature vectors, the "face_recognition" library can perform face recognition by comparing the features of a query face with the features of known faces stored in a database. It calculates the similarity or distance between the feature vectors and determines the closest matches. You can specify a threshold to define the level of similarity required for a positive recognition.

#. Usage: To use the library, you typically load an image or video frame, detect faces, align the faces (optional), and then extract and compare the face features for recognition. The library provides easy-to-use functions and classes to perform these tasks, allowing you to integrate face recognition capabilities into your Python applications.

.. note::
    It's worth noting that the "face_recognition" library is built on top of popular deep learning frameworks like dlib and OpenCV. It provides a high-level interface to simplify face recognition tasks and abstracts away the complexities of model training and implementation.

Remember that face recognition accuracy can be influenced by factors such as image quality, variations in lighting and pose, and the number of training examples available for known faces. Therefore, it's important to consider these factors and perform proper testing and fine-tuning to achieve optimal results in your specific use case.

dlib
-------------------

Why do we need dlib
-------------------

dlib is a dependency that provides essential functionality for face detection and facial landmark estimation. The "face_recognition" library builds upon dlib's capabilities to offer a higher-level interface specifically for face recognition tasks.

Dlib is a C++ library that contains various machine learning algorithms and tools, including the implementation of the "HOG" (Histogram of Oriented Gradients) feature descriptor and the "SVM" (Support Vector Machine) classifier. These components are used by the "face_recognition" library to perform face detection, facial landmark detection, and face alignment.

Here's a breakdown of dlib's role in the face recognition process:

#. Face Detection: The face detection component in dlib utilizes the "HOG" feature descriptor and the "SVM" classifier to detect faces in images. It searches for patterns in image gradients to identify regions likely to contain faces.

#. Facial Landmark Detection: Dlib provides a facial landmark estimation model, which is trained to identify key facial landmarks such as the eyes, nose, and mouth. These landmarks are crucial for face alignment and accurately extracting facial features.

#. Face Alignment: Using the detected facial landmarks, dlib performs face alignment by applying geometric transformations to normalize the face's position, scale, and orientation. Face alignment helps ensure that subsequent face recognition tasks are robust to variations in pose and improve the accuracy of feature extraction.

#. The "face_recognition" library abstracts the usage of dlib by providing a simpler, user-friendly interface that allows developers to focus on the face recognition tasks rather than dealing with low-level details. It leverages the face detection and facial landmark estimation capabilities of dlib to provide a higher-level API for face recognition and feature extraction.

In summary, dlib plays a critical role in providing the underlying functionality for face detection, facial landmark detection, and face alignment in the "face_recognition" library. It enables accurate and efficient face recognition by handling the low-level details of these tasks, allowing developers to work with face recognition in a more accessible manner using the "face_recognition" library.

Install
--------------------------------

To install dlib, you need to ensure that you meet the following specifications:

#. Operating System: dlib is compatible with Windows, macOS, and Linux operating systems.

#. Python Version: dlib works with Python 2.7 or Python 3.x versions.

#. Compiler: You need a C++ compiler to build and install dlib. For Windows, you can use Microsoft Visual C++ (MSVC) or MinGW. On macOS, Xcode Command Line Tools are required. On Linux, the GNU C++ Compiler (g++) is typically used.

#. Dependencies: dlib relies on a few external dependencies, including Boost and CMake. These dependencies need to be installed beforehand to successfully build dlib.


For windows installation
--------------------------------
The official installation instructions are found here https://github.com/ageitgey/face_recognition/issues/175#issue-257710508

#. To install the library, first you need to install the dlib library
    #. Installation instructions are here
    #. Download CMake windows installer from here
        #. While installing CMake select Add CMake to the system PATH to avoid any error in the next steps.
    #. Install VIsual C++ download here
    #.
    .. code-block::bash

        pip install cmake

    #.
    .. code-block::bash

        pip install dlib
    #.
    .. code-block::bash

        pip install face_recognition


Resources
--------------------------------
*. https://pypi.org/project/face-recognition/#description
*. https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf
*. https://github.com/ageitgey/face_recognition/issues/175#issue-257710508

Integration in DroneBuddy
~~~~~~~~~~~~~~~~~~~~~~~

Add faces to the memory
--------------------------------
 In order to proceed with the face recognition , the algorithm needs encoding s of the known faces. The library has a method that is specifically designed to add these faces to the memory.

.. code-block:: python

     engine_configs = EngineConfigurations({})
     image = cv2.imread('test_clear.jpg')
     engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECC, engine_configs)
     result = engine.remember_face(image, "Jane")



You can check if the images and names are added to the library by simply going to the location where the library is installed.

.. code-block:: bash

    newvenv/Lib/site-packages/dronebuddylib/atoms/resources


Recognize faces
--------------------------------
To use the recognition feature you can use the find_all_the_faces method by simply feeding the frame to it. The show_feed variable refers to whether to show the video feed in a new window. The default settings for this is false. The method returns a list of names of the people in the frame, if not recognized it will return unknown.

.. code-block:: python

     engine_configs = EngineConfigurations({})
     image = cv2.imread('test_jane.jpg')
     engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECC, engine_configs)
     result = engine.recognize_face(image)




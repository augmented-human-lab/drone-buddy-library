face_recognition Recognition
========================


Face-recognition is an open-source Python library that provides face detection, face alignment, and face recognition capabilities.
The official documentation can be found `here <https://github.com/ageitgey/face_recognition>`_

Installation
-------------

The face_recognition requires the following pre-requisites
#.  dlib


dlib Installation
~~~~~~~~~~~~~~~~~
        To install dlib, you need to ensure that you meet the following specifications:

        Operating System: dlib is compatible with Windows, macOS, and Linux operating systems.

        Python Version: dlib works with Python 2.7 or Python 3.x versions.

        Compiler: You need a C++ compiler to build and install dlib. For Windows, you can use Microsoft Visual C++ (MSVC) or MinGW. On macOS, Xcode Command Line Tools are required. On Linux, the GNU C++ Compiler (g++) is typically used.

        Dependencies: dlib relies on a few external dependencies, including Boost and CMake. These dependencies need to be installed beforehand to successfully build dlib.

Windows
^^^^^^^
           The official installation instructions are found `here <https://github.com/ageitgey/face_recognition/issues/175#issue-257710508>`_

            -   To install the library, first you need to install the dlib library

            Installation instructions are here

            #.  Download CMake windows installer from `here <https://cmake.org/download/>`_
            #.  While installing CMake select Add CMake to the system PATH to avoid any error in the next steps.
            #.   Install Visual C++ , if not installed previously


            Then run the following commands to install the face_recognition
            #.  cmake installation
            .. code-block::

                pip install cmake

            #.  dlib installation

            .. code-block::

                pip install dlib


            #.  face_recognition


mac OS installation
^^^^^^^^^^^^^^^^^^^

           mac installation is pretty straightforward.



.. code-block::

    pip install dronebuddylib[FACE_RECOGNITION]


Usage
-------------

Add faces to the memory
^^^^^^^^^^^^^^^^^^^^^^^

In order to proceed with the face recognition , the algorithm needs encoding s of the known faces. The library has a method that is specifically designed to add these faces to the memory.

.. code-block:: python

    engine_configs = EngineConfigurations({})
    image = cv2.imread('test_clear.jpg')
    engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECC, engine_configs)
    result = engine.remember_face(image, "Jane")

You can check if the images and names are added to the library by simply going to the location where the library is installed.

.. code-block::python
    venv/Lib/site-packages/dronebuddylib/atoms/resources



#.  SPEECH_RECOGNITION_VOSK_LANGUAGE_MODEL_PATH - This is the path to the model that you have downloaded. This is a compulsory parameter if you are using any other language. If this is not provided, the default model will be used. The default model is the english model ( vosk-model-small-en-us-0.15 ). Vosk supported languages can be found `here <https://alphacephei.com/vosk/models>`_.

Recognize faces
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
    engine_configs = EngineConfigurations({})
    image = cv2.imread('test_jane.jpg')
    engine = FaceRecognitionEngine(FaceRecognitionAlgorithm.FACE_RECC, engine_configs)
    result = engine.recognize_face(image)


Output
-------------

The output will be a list of names, if no people are spotted in the frame empty list will be returned. If people are spotted but not recognized, 'unknown' will be added as a list item.

Resources
---------
-   https://pypi.org/project/face-recognition/#description
-   https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf
-   https://github.com/ageitgey/face_recognition/issues/175#issue-257710508

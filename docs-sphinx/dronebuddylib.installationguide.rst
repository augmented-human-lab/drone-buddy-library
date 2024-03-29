Installation Guide
==================

Introduction
------------


DroneBuddy envisions empowering everyone with the ability to personally program their intelligent drones, enriching them with desired features. At its core, DroneBuddy offers a suite of fundamental building blocks, enabling users to seamlessly integrate these elements to bring their drone to flight.

Functioning as an intuitive interface, DroneBuddy simplifies complex algorithms, stripping away the intricacies to offer straightforward input-output modalities. This approach ensures that users can accomplish their objectives efficiently, without getting bogged down in technical complexities. With DroneBuddy, the focus is on user-friendliness and ease of use, making drone programming accessible and hassle-free.


Installation
------------

DroneBuddy behaves as any other python library. You can find the library at https://pypi.org/project/dronebuddylib/ and install using pip.

.. code-block:: bash

    pip install dronebuddylib


The installation of DroneBuddy needs the following prerequisites.

#.  Python 3.9 or higher
#.  Compatible pip version

.. note::

    Note that running `pip install dronebuddylib ` will only install the drone buddy library, with only the required dependencies.
    which are
    #.  requests -
    #.  numpy       - required by
    #.  cython      -
    #.  setuptools
    #.  packaging
    #.  pyparsing

...

Installation of Features
~~~~~~~~~~~~~~~~~~~~~~~~
Each and every feature and it's required dependencies can be installed by the following code snippet

.. code-block::

    pip install dronebuddylib[FEATURE_NAME]

Intent Recognition Module Installation
-------------------------------------

.. toctree::
   :maxdepth: 3

   dronebuddylib.intentrecognitioninstallation



Voice Recognition Module Installation
-------------------------------------

.. toctree::
   :maxdepth: 3

   voice_recogntion_installation_guide




Face Recognition Module Installation
-------------------------------------

.. toctree::
   :maxdepth: 3

   face_recogntion_installation_guide



Object Detection Module Installation
-------------------------------------

.. toctree::
   :maxdepth: 3

   object_detection_installation_guide




Voice Generation Module Installation
-------------------------------------

.. toctree::
   :maxdepth: 3

   voice_generation_installation_guide






Text Recognition Module Installation
-------------------------------------

.. toctree::
   :maxdepth: 3

   text_recognition_installation_guide







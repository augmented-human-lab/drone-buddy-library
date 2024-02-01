Snips- NLU Module
======================

Installation
-------------

To install SNIPS NLU Integration run the following snippet, which will install the required dependencies
The official documentation can be found [here](https://snips-nlu.readthedocs.io/en/latest/)

The SNIPS NLU requires the following pre-requisites
    #.  Rust - [Official documentation](https://www.rust-lang.org/tools/install)
    #.  Visual C++ Build Tools
    #.  setuptools-rust



Rust Installation
~~~~~~~~~~~~~~~~~
        Rust is required to install the intent recognition library that is needed to understand the commands given to the drone.

        You can download Rust from [here](https://www.rust-lang.org/tools/install)

        The steps will differ from Windows to mac, so follow the steps on the website.

Windows
^^^^^^^
            Windows installation is pretty straightforward

mac OS installation
^^^^^^^^^^^^^^^^^^^

            If you get a permission error while installing , follow the steps in the stackoverflow [answer](https://stackoverflow.com/questions/45899815/could-not-write-to-bash-profile-when-installing-rust-on-macos-sierra ) .

            Give a try using this not using sudo:

            .. code-block:: bash

                curl https://sh.rustup.rs -sSf | sh -s – –help


            If that works then probably you could try:

            .. code-block:: bash

                curl https://sh.rustup.rs -sSf | sh -s – –no-modify-path


            If the command with the –no-modify-path option works, you’ll have to manually update .bash_profile to include it in your path:

            .. code-block:: bash

                source ~/.cargo/env

            Now you are all ready to install Setuptools-rust.

Visual C++ Build Tools
~~~~~~~~~~~~~~~~~
        setuptools-rust  requires the Visual C++ Build Tools to be installed on your system. You can download and install them from the Microsoft website [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/ ).


setuptools-rust
~~~~~~~~~~~~~~~~~

        In your virtual environment

        .. code-block::

            pip install setuptools-rust

        This will download and install setuptools-rust and its dependencies. Note: If you encounter any errors during the installation process, try upgrading pip to the latest version by running pip install –upgrade pip before installing setuptools-rust. You may also need to add Rust and the Visual C++ Build Tools to your system’s PATH environment variable.

.. code-block::

    pip install dronebuddylib[INTENT_RECOGNITION_SNIPS]


Usage
-------------

The Snips integration module have the following configurations as optional

#.  INTENT_RECOGNITION_SNIPS_NLU_DATASET_PATH          - Current SNIPs uses a default data set that is created by the owners, which supports basic drone controls, if you need to retrain the NLU for your data set, pass the path of the data set here. Refer to the original documentation for tips to create the data set
#.  INTENT_RECOGNITION_SNIPS_LANGUAGE_CONFIG           - The language your new data set is in, supported languages can be found [here](https://snips-nlu.readthedocs.io/en/latest/installation.html#language-resources) and follow the steps mentioned here for proper setup

A sample data set can be found at `dornebuddylib/atoms/resources/intentrecognition/`

Intents
^^^^^^^

        Intents define what the intended objective of  a phrase is. You can define the intent according to the needs of the drone.
        Currently the defined intents are defined in the enum class DroneCommands to ease the programming. This can be overridden anytime.

        Defining Intents


        .. code-block:: yml

            # takeoff intent
            ---
            type: intent
            name: TAKE_OFF
            slots:
             - name: DroneName
               entity: DroneName
             - name: address
               entity: address_entity
            utterances:
             - Take off the [DroneName](quadcopter) from the ground.
             - "[address](sammy) Launch the [DroneName](hexacopter) and make it fly."
             - hey [address](sammy), get in the air.
             - Lift off from the launchpad.
             - take off
             - hey [address](sammy),launch
             - fly
             - "[address](sammy) can you please take off"
             - hey [address](sammy), can you please launch
             - can you please fly
             - hey [address](sammy), can you please take off [DroneName](this)

        Utterances refer to the training sentences that should be used to train the specific intent.  Generally these should cover a lot of possible, then the intent recognition will give better accuracy.

        You can cover all the intents that you need to cover when programming the drone.

Entities
^^^^^^^^

        Entities refer to the entities that need to be recognized in the conversation. For an example these can be distance, names, locations, directions.
        Defining entities

        .. code-block:: yml

            # address Entity
            ---
            type: entity # allows to differentiate between entities and intents files
            name: address_entity # name of the entity
            values:
             - [ sammy, semi , semmi , sami , sammi , semmi ] # entity value with a synonym


            [name](Sammy) can you move [distance](10m) to your [direction](right)


        In the default case the entity is defined as the name to address the drone. Entity is defined as address_entity, this can be retrieved in the recognized intent.

Train the NLU with new data set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Train the NLU If you are planning to override the the existing data set, you can simply create dataset.yaml , Modify the paths in the following command to generate the json file.

    .. code-block::
        snips-nlu generate-dataset en  C:\Users\janedoe\projects\DroneBuddy\drone-buddy-library\dronebuddylib\resources\intentrecognition\dataset.yaml > C:\Users\janedoe\projects\DroneBuddy\drone-buddy-library\dronebuddylib\resources\intentrecognition\dataset.json


Code Example
-------------

.. code-block:: python

    engine_configs = EngineConfigurations({})
    intent_engine = IntentRecognitionEngine(IntentRecognitionAlgorithm.SNIPS_NLU, engine_configs)
    result = intent_engine.recognize_intent("what can you see?")



Output
-------------

The output will be given in the following json format

.. code-block:: json

    {
        "intent": "RECOGNIZED_INTENT",
        "entities": [
            {
                "entity_type" : "",
                "value" : ""
            }
        ],
        "confidence": 1.0,
        "addressed_to": false
    }


Where
    - intent - Recognized intent
    - entity - Recognized entities
    - confidence  - Confidence of the recognition


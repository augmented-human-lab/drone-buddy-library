Supported models
==========

SNIPS NLU
~~~~~~~~~~~~~~~~~~~~~~~

Snips NLU (Natural Language Understanding) is an open-source library designed to perform intent recognition and slot filling, two essential tasks in natural language processing. It allows computers to understand the meaning and extract relevant information from user queries or commands.

#. Training Data: Snips NLU requires training data to learn how to understand and process user queries. Training data consists of labeled examples, including user queries and their corresponding intents and slots. Intents represent the user's intention, while slots capture specific pieces of information within the query.

#. Intent Recognition: Snips NLU uses machine learning algorithms to train a model on the provided training data. During training, the model learns to recognize different intents by analyzing the patterns and relationships between the words or features in the queries and their corresponding intents. The trained model can then predict the intent of new, unseen queries.

#. Slot Filling: In addition to intent recognition, Snips NLU also performs slot filling. Slot filling involves identifying and extracting specific information or parameters (slots) from the user's query. For example, in the query "Book a table for two at 7 PM," the slots could be "table" (slot type: restaurant table) and "time" (slot type: time). Snips NLU learns to recognize and extract these slots based on the patterns observed in the training data.

#. Model Deployment: Once the model is trained, it can be deployed and integrated into your application or system. Snips NLU provides a simple API that allows you to send user queries to the model and receive the recognized intent and extracted slots as the output.

#. Intent Recognition and Slot Filling in Action: When a user query is sent to the deployed Snips NLU model, it processes the text and predicts the intent based on the learned patterns. Additionally, it identifies and extracts relevant slots from the query, providing structured information about the user's request.

#. Output Generation: The recognized intent and extracted slots are generated as output, enabling your application to understand the user's intention and access the specific information provided in the query. This output can be further processed to trigger appropriate actions or provide relevant responses based on the recognized intent and slots.

Snips NLU is designed to be flexible and customizable, allowing you to train models specific to your domain or application. It provides tools to annotate training data, train the models, and evaluate their performance.

By using Snips NLU, you can incorporate natural language understanding capabilities into your applications, such as chatbots, voice assistants, or any system that requires understanding and processing of user queries.

Integration in DroneBuddy
---------------------------------

Install setuptools-rust
------------------------------------------------------------------

To install setuptools-rust on Windows, you can follow these steps:

#. Install Rust: setuptools-rust requires Rust to be installed on your system. You can download and install Rust from the official website at https://www.rust-lang.org/tools/install .
#. Install Visual C++ Build Tools: setuptools-rust also requires the Visual C++ Build Tools to be installed on your system. You can download and install them from the Microsoft website at https://visualstudio.microsoft.com/visual-cpp-build-tools/.
#. Install Python: If you haven't already, you need to install Python on your system. You can download and install the latest version of Python from the official website at https://www.python.org/downloads/windows/.
#. Open a command prompt: Open a command prompt by pressing the Windows key + R, typing "cmd" in the Run dialog box, and pressing Enter.
#. Install setuptools-rust: In the command prompt, navigate to the directory where you want to install setuptools-rust, and run the following command:

.. code-block:: bash

    pip install setuptools-rust


This will download and install setuptools-rust and its dependencies.
Note: If you encounter any errors during the installation process, try upgrading pip to the latest version by running pip install --upgrade pip before installing setuptools-rust. You may also need to add Rust and the Visual C++ Build Tools to your system's PATH environment variable.


Install required dependencies for Snips NLU
------------------------------------------------------------------
The easiest way to install the modules required for SNIPS NLU is to use the following command.

.. code-block:: bash

    pip install dronebuddylib[INTENT_RECOGNITION_SNIPS]

This will take care of the dependencies and install the required modules.


Install Snips NLU
------------------------------------------------------------------

Or you can manually install the required modules by using the following commands.

.. code-block:: bash

 pip install snips-nlu



Create data set
---------------------------------

A sample training set is already created with the most common use cases for the drone. It can be accessed in the location;

.. code-block:: bash

 dornebuddylib/atoms/resources/intentrecognition/


Standard training set contains a defined set of entities and intents.

Intents
---------------------------------

Intents define what the intended objective of  a phrase is. You can define the intent according to the needs of the drone.
Currently the defined intents are defined in the enum class DroneCommands to ease the programming. This can be overridden anytime.

Defining Intents


.. code-block:: json

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
---------------------------------

Entities refer to the entities that need to be recognized in the conversation. For an example these can be distance, names, locations, directions.
Defining entities

.. code-block:: json

    # address Entity
    ---
    type: entity # allows to differentiate between entities and intents files
    name: address_entity # name of the entity
    values:
     - [ sammy, semi , semmi , sami , sammi , semmi ] # entity value with a synonym


    [name](Sammy) can you move [distance](10m) to your [direction](right)


In the default case the entity is defined as the name to address the drone. Entity is defined as address_entity, this can be retrieved in the recognized intent.

The response will be as follows

.. code-block:: json

    {
      "input": "sammy can you move 10m to your right? ",
      "intent": {
        "intentName": "RIGHT",
        "probability": 1
      },
      "slots": [
        {
          "range": {
            "start": 0,
            "end": 5
          },
          "rawValue": "sammy",
          "value": {
            "kind": "Custom",
            "value": "sammy"
          },
          "entity": "address_entity",
          "slotName": "address"
        },
        {
          "range": {
            "start": 19,
            "end": 22
          },
          "rawValue": "10m",
          "value": {
            "kind": "Custom",
            "value": "10m"
          },
          "entity": "distance_entity",
          "slotName": "distance"
        }
      ]
    }



This feature was introduced to reduce the noise of the voice recognition. When the drone is tested in a noisy environment the drone responds to every conversation. In order to stop this you can enable the activation phrase feature, which enables you to command the drone by addressing the drone directly by its name.
The default name is sammy, which was selected as the probability of it being misrecognized is comparatively lower.
If you need to change the name, you need to alter the training data set according to your needs.

The method is_addressed_to_drone can be used to decide whether the drone is being addressed or not.
Train the NLU
If you are planning to override the the existing data set, you can simply create dataset.yaml ,
Modify the paths in the following command to generate the json file.


.. code-block:: bash

    snips-nlu generate-dataset en  C:\Users\janedoe\projects\DroneBuddy\drone-buddy-library\dronebuddylib\resources\intentrecognition\dataset.yaml > C:\Users\janedoe\projects\DroneBuddy\drone-buddy-library\dronebuddylib\resources\intentrecognition\dataset.json


Using the NLU
--------------------------

The NLU can be used to recognize the intent and the slots of the recognized intent. The following code snippet can be used to recognize the intent and the slots.

.. code-block:: python

      def test_object_recognition_engine_with_snips(self):
        engine_configs = EngineConfigurations({})
        engine = IntentRecognitionEngine(IntentRecognitionAlgorithm.SNIPS_NLU, engine_configs)
        result = engine.recognize_intent("find the chair")


Chat GPT
~~~~~~~~~~~~~~~~~~~~~~~

DroneBuddy is integrating ChatGPT for intent resolution. This section explores the role of ChatGPT in DroneBuddy, highlighting its capabilities, features, and integration process. For more comprehensive details about ChatGPT, refer to OpenAI's official documentation.

#. Language Understanding: ChatGPT is adept at interpreting natural language inputs in DroneBuddy. It analyzes user queries or commands to discern the underlying intents, essential for providing accurate responses or actions.

#. Contextual Awareness: A key strength of ChatGPT in DroneBuddy is its ability to maintain context throughout a conversation. This ensures understanding of follow-up queries or references to previous parts of the dialogue, enhancing the user experience.

#. Response Generation: In DroneBuddy, ChatGPT is tasked with generating human-like, coherent responses that are contextually appropriate and informative, based on the user's intent.

#. Continuous Learning: While ChatGPT comes pre-trained on extensive textual data, DroneBuddy can fine-tune it for specific domains or applications, improving its effectiveness and relevance.

Install required dependencies for chat GPT
------------------------------------------------------------------
The easiest way to install the modules required for SNIPS NLU is to use the following command.

.. code-block:: bash

    pip install dronebuddylib[INTENT_RECOGNITION_GPT]

This will take care of the dependencies and install the required modules.

Install Snips NLU
------------------------------------------------------------------

Or you can manually install the required modules by using the following commands.

.. code-block:: bash

    pip install openai

    pip install tiktoken

Integration in DroneBuddy
-------------------------

#. Setup and Configuration: Integrating ChatGPT in DroneBuddy involves setting up an environment that interfaces with OpenAI's API, including obtaining and configuring API keys.

#. User Input Processing: DroneBuddy captures user inputs, which can be text or speech (converted to text), and sends them to ChatGPT for processing.

#. Handling Responses: Upon receiving a response from ChatGPT, DroneBuddy manages this output accordingly. This may involve displaying it to the user, triggering specific functions, or logging for analysis.

#. Context Management: For conversational continuity, DroneBuddy manages the context across interactions, tracking conversation history and providing it to ChatGPT for each new request.

Example of ChatGPT Integration in DroneBuddy
---------------------------------------------

Here’s a hypothetical example of how ChatGPT could be integrated from DroneBuddy for intent resolution:

.. code-block:: python

    engine_configs = EngineConfigurations({})
    engine_configs.add_configuration(Configurations.INTENT_RECOGNITION_OPEN_AI_TEMPERATURE, "0.7")
    engine_configs.add_configuration(Configurations.INTENT_RECOGNITION_OPEN_AI_MODEL, "gpt-3.5-turbo-0613")
    engine_configs.add_configuration(Configurations.INTENT_RECOGNITION_OPEN_AI_LOGGER_LOCATION,
                                         "C:\\Users\\Public\\projects\\drone-buddy-library\\dronebuddylib\\atoms\\intentrecognition\\resources\\stats\\")
    engine_configs.add_configuration(Configurations.INTENT_RECOGNITION_OPEN_AI_API_KEY,
                                         "sk-*****************************************")
    engine_configs.add_configuration(Configurations.INTENT_RECOGNITION_OPEN_AI_API_URL,
                                         "https://api.openai.com/v1/chat/completions")
    engine = IntentRecognitionEngine(IntentRecognitionAlgorithm.CHAT_GPT, engine_configs)
    result = engine.recognize_intent("take a picture of a chair")


This code snippet demonstrates the process of sending a user query to ChatGPT and receiving a contextually relevant response, which DroneBuddy can then use in various ways.

Important Considerations
------------------------
.. important::
    It's crucial to remember that ChatGPT, as implemented in DroneBuddy, generates responses based on learned data patterns and probabilities. Therefore, its output might not always be perfectly accurate or suitable for every situation. Continuous monitoring and occasional fine-tuning are recommended to ensure the system aligns with DroneBuddy's specific needs and provides optimal results.
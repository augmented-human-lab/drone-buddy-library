GPT Integration Module
======================

Installation
-------------

To install Open Ai Integration run the following snippet, which will install the required dependencies

.. code-block::

    pip install dronebuddylib[INTENT_RECOGNITION_GPT]


Usage
-------------

The OpenAi integration module requires the following configurations to function

#.  INTENT_RECOGNITION_OPEN_AI_API_URL          - URL provided by openai to send the requests, this can be found in [openai apidoc](https://platform.openai.com/docs/api-reference)
#.  INTENT_RECOGNITION_OPEN_AI_MODEL            - The model you need to use, the model list can be found [here](https://platform.openai.com/docs/models)
#.  INTENT_RECOGNITION_OPEN_AI_API_KEY          - API key provided to you by openai found in the [API key page](https://platform.openai.com/account/api-keys)
#.  INTENT_RECOGNITION_OPEN_AI_TEMPERATURE      - OpenAI's models adjusts the randomness of responses, with lower values leading to more predictable text and higher values increasing creativity and variability.
#.  INTENT_RECOGNITION_OPEN_AI_LOGGER_LOCATION  - A location in your local folder to store the logs of the conversation history

Code Example
-------------

.. code-block:: python

    engine_configs = EngineConfigurations({})
    engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_TEMPERATURE, "0.7")
    engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_MODEL, "gpt-3.5-turbo-0613")
    engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_LOGGER_LOCATION,
                                         "C:\\Users\\Public\\projects\\drone-buddy-library\\dronebuddylib\\atoms\\intentrecognition\\resources\\stats\\")
    engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_API_KEY,
                                         "sk-*********** YOUR API KEY *****************")
    engine_configs.add_configuration(AtomicEngineConfigurations.INTENT_RECOGNITION_OPEN_AI_API_URL,
                                         "https://api.openai.com/v1/chat/completions")
    intent_engine = IntentRecognitionEngine(IntentRecognitionAlgorithm.CHAT_GPT, engine_configs)

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

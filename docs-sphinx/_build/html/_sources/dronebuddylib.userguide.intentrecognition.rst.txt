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


Chat GPT
~~~~~~~~~~~~~~~~~~~~~~~

DroneBuddy is integrating ChatGPT for intent resolution. This section explores the role of ChatGPT in DroneBuddy, highlighting its capabilities, features, and integration process. For more comprehensive details about ChatGPT, refer to OpenAI's official documentation.

#. Language Understanding: ChatGPT is adept at interpreting natural language inputs in DroneBuddy. It analyzes user queries or commands to discern the underlying intents, essential for providing accurate responses or actions.

#. Contextual Awareness: A key strength of ChatGPT in DroneBuddy is its ability to maintain context throughout a conversation. This ensures understanding of follow-up queries or references to previous parts of the dialogue, enhancing the user experience.

#. Response Generation: In DroneBuddy, ChatGPT is tasked with generating human-like, coherent responses that are contextually appropriate and informative, based on the user's intent.

#. Continuous Learning: While ChatGPT comes pre-trained on extensive textual data, DroneBuddy can fine-tune it for specific domains or applications, improving its effectiveness and relevance.


Important Considerations
------------------------
.. important::
    It's crucial to remember that ChatGPT, as implemented in DroneBuddy, generates responses based on learned data patterns and probabilities. Therefore, its output might not always be perfectly accurate or suitable for every situation. Continuous monitoring and occasional fine-tuning are recommended to ensure the system aligns with DroneBuddy's specific needs and provides optimal results.
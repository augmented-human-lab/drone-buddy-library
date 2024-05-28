Available Agents
==========


Object Identifier Agent
~~~~~~~~~~~~~~~~~~~~~~~

Overview
--------

The `ObjectIdentifierAgentImpl` class is designed to implement an object identifier agent using a Large Language Model (LLM). This class enables the functionality to remember objects and identify objects in images, leveraging advanced AI capabilities.

Initialization
--------------

To set up an instance of `ObjectIdentifierAgentImpl`, you need to provide several parameters including the API key for accessing the LLM, the model name, and optionally, the temperature setting for the model's responses and a location for logging information. Upon initialization, the system prompt specific to object identification is set up to guide the LLM’s behavior.

Key Methods
-----------

1. **get_agent_name**

   This method retrieves the name of the LLM agent, which is useful for identifying which specific agent instance is being referred to in a multi-agent setup.

2. **get_agent_description**

   This method provides a description of the LLM agent, offering more detailed information about the agent’s purpose and functionality.

System Prompt
-------------

The system prompt for the `ObjectIdentifierAgentImpl` is pre-defined to ensure the LLM understands and executes its tasks effectively. The prompt sets the context and instructions for the LLM to follow when interacting with the agent.

The process includes two main instructions:

1. **Remembering Objects**: When an instruction to remember an object is given, along with an image of the object, the LLM is expected to store the object in its memory and acknowledge the action. The acknowledgment includes a status indicating success or failure and a descriptive message.

2. **Identifying Objects**: When an instruction to identify objects is provided with an image, the LLM processes the image to identify all objects. The output is structured as a list of objects, each with details such as the class it belongs to, its name (if it was previously remembered), a description, and a confidence score indicating the reliability of the identification.

Process Summary
---------------

- **Initialization**: Set up the agent with the necessary parameters and define the system prompt for object identification.
- **Remember Instruction**: Accepts an image and an object name, stores the object in memory, and returns a status and message.
- **Identify Instruction**: Accepts an image, identifies objects, and returns a detailed list of identified objects with their attributes.

This documentation explains the core functionality and workflow of the `ObjectIdentifierAgentImpl` class, providing an understanding of how it operates within the context of using an LLM for object identification tasks.

Image Describer Agent
~~~~~~~~~~~~~~~~~~~~~~~

Overview
--------

The `ImageDescriberAgentImpl` class is designed to implement an image describer agent using a Large Language Model (LLM). This class provides functionalities to describe images, particularly to assist visually impaired individuals by offering detailed descriptions of objects in images.

Initialization
--------------

To set up an instance of `ImageDescriberAgentImpl`, several parameters need to be provided, including the API key for accessing the LLM, the model name, and optionally, the temperature setting for the model's responses and a location for logging information. Upon initialization, a system prompt specific to image description is configured to guide the LLM’s behavior.

Key Methods
-----------

1. **get_agent_name**

   This method retrieves the name of the LLM agent, which is useful for identifying which specific agent instance is being referred to in a multi-agent setup.

2. **get_agent_description**

   This method provides a description of the LLM agent, offering more detailed information about the agent’s purpose and functionality.

3. **get_result**

   This method gets the description result from the LLM and formats it into an `ImageDescriberResults` object. The method processes the LLM's response, converting it into a structured format that includes the object's name, a detailed description, and a confidence score.

System Prompt
-------------

The system prompt for the `ImageDescriberAgentImpl` is pre-defined to ensure the LLM understands and executes its tasks effectively. The prompt sets the context and instructions for the LLM to follow when interacting with the agent.

The process includes a primary instruction:

1. **Describe Instruction**: When an image and the instruction to describe it are given, the LLM provides a detailed explanation of the object within the image. The output is structured in a format that includes the object's name, a full description for the person to listen to, and a confidence score indicating the reliability of the description.

Process Summary
---------------

- **Initialization**: Set up the agent with the necessary parameters and define the system prompt for image description.
- **Describe Instruction**: Accepts an image and returns a detailed description, structured to assist visually impaired individuals.
- **Result Formatting**: Converts the LLM's response into a structured format, including the object's name, description, and confidence score.

This documentation explains the core functionality and workflow of the `ImageDescriberAgentImpl` class, providing an understanding of how it operates within the context of using an LLM for image description tasks.


Image Validator Agent
~~~~~~~~~~~~~~~~~~~~~~~

Overview
--------

The `ImageValidatorAgentImpl` class is designed to implement an image validator agent using a Large Language Model (LLM). This class provides functionalities to validate images for object identification, ensuring that the images are suitable for future recognition tasks.

Initialization
--------------

To set up an instance of `ImageValidatorAgentImpl`, several parameters need to be provided, including the API key for accessing the LLM, the model name, and optionally, the temperature setting for the model's responses and a location for logging information. Upon initialization, a system prompt specific to image validation is configured to guide the LLM’s behavior.

Key Methods
-----------

1. **get_agent_name**

   This method retrieves the name of the LLM agent, which is useful for identifying which specific agent instance is being referred to in a multi-agent setup.

2. **get_agent_description**

   This method provides a description of the LLM agent, offering more detailed information about the agent’s purpose and functionality.

3. **get_result**

   This method gets the validation result from the LLM and formats it into an `ImageValidatorResults` object. The method processes the LLM's response, converting it into a structured format that includes the object type, validation status, a description, and specific instructions if the image needs improvements.

System Prompt
-------------

The system prompt for the `ImageValidatorAgentImpl` is pre-defined to ensure the LLM understands and executes its tasks effectively. The prompt sets the context and instructions for the LLM to follow when interacting with the agent.

The process includes a primary instruction:

1. **Validate Instruction**: When an image and the instruction to validate it are given, the LLM assesses whether the image is suitable for future recognition. The output is structured in a format that includes the object type, a validation status (true if the image is good enough, false if not), a description of the object, and specific instructions for improving the image if necessary.

Process Summary
---------------

- **Initialization**: Set up the agent with the necessary parameters and define the system prompt for image validation.
- **Validate Instruction**: Accepts an image and returns a validation result, structured to assist in determining the suitability of the image for recognition.
- **Result Formatting**: Converts the LLM's response into a structured format, including the object type, validation status, description, and improvement instructions.

This documentation explains the core functionality and workflow of the `ImageValidatorAgentImpl` class, providing an understanding of how it operates within the context of using an LLM for image validation tasks.




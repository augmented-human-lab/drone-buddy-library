"""This is the doc string for the utils1 file where we can say things about the python module.add()
We can write long text if we want.

* topic 1
* topic 2
"""

import openai as openai

import dronebuddylib.configurations.config as config

openai.api_key = config.openai_api_token

# Set up the API parameters
model_engine = "text-davinci-002"
max_tokens = 50
temperature = 0.5
n = 1
stop = None


def prompt_chatgpt(prompt):
    """
    Generates a response using the OpenAI GPT model based on the provided prompt.

    Args:
        prompt (str): The input prompt to initiate the conversation.

    Returns:
        str: The generated response from the OpenAI GPT model.

    Example:
        response = prompt_chatgpt("Tell me a joke.")
        print(response)
    """
    # Call the OpenAI API
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        stop=stop,
    )

    message = response.choices[0].text.strip()
    return message

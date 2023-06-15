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

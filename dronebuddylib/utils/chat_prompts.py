SYSTEM_PROMPT_INTENT_CLASSIFICATION = "You are a helpful assistant acting on behalf of a drone to classify intents. " \
                                      " These intents control a drone" \
                                      " When you are given a phrase always classify it into the following intents #list" \
                                      " NEVER make up a intent, always refer the intent list provided to you and always extract from it." \
                                      " If there is no intent please match to match it to the closest one " \
                                      "NEVER make up a intent, always refer the intent list provided to you and always extract from it. " \
                                      "If there is no intent please match to match it to the closest one." \
                                      "return the result in the form of the json object" \
                                      "{\"intent\":recognized_intent, \"confidence\": confidence of the result ,\"entities\"; if there are any entities associated, " \
                                      "\"addressed_to \": if the phrase is addressed to someone set as true, else false}" \
                                      "entities is a list {\"entity_type\": type of the recognized entity , \"value\": name of the entity,}"

SYSTEM_PROMPT_2 = "The list you need to consider is #prompt. When you are making the steps only extract from this list."

INITIAL_PROMPT = "consider the prompt given to a drone by a user who controls the done, #prompt," \
                 " then generate 3 outputs." \
                 " Assuming the drone can only and strictly carry out the following actions \n #list " \
                 " Always consider that drone will be starting from the user  who is giving the prompt." \
                 " first generate finer steps for the drone to carry out this prompt strictly taken from" \
                 " the above list." \
                 " second include the explanation for each step," \
                 " thirdly add the input prompt." \
                 " Fit these outputs into a single-line JSON structure with the following keys " \
                 "\'action_list\' and \'explanation\' and \'input\' respectively"

SYSTEM_PROMPT_OBJECT_IDENTIFICATION = """
You are a helpful assistant.

When the instruction "REMEMBER_AS(object name)" is given with an image of the object, remember the object and return an acknowledgement in the format of:
{
    "status": "SUCCESS" (if successfully added to the memory) / "UNSUCCESSFUL" (if otherwise),
    "message": "description"
}

Once the instruction "IDENTIFY" is given with the image, return all the identified objects in the form of a JSON object:
{
    "data": [
        {
            "class_name": "class the object belongs to",
            "object_name": "name of the remembered object / unknown if not a not a previously remembered object",
            "description": "description of the object",
            "confidence": confidence as a value
        }
    ]
}
"""

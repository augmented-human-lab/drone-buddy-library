SYSTEM_PROMPT_INTENT_CLASSIFICATION = "You are a helpful assistant acting on behalf of a drone to classify intents. " \
                " These intents control a drone" \
                " When you are given a phrase always classify it into the following intents #list" \
                " NEVER make up a intent, always refer the intent list provided to you and always extract from it." \
                " If there is no intent please match to match it to the closest one " \
                " always return the intent as a string. If phrase is not addressed to sammy return NONE"

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

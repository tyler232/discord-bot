import random
import discord
import json
import handler
import re

def load_json(file):
    with open(file) as client_response:
        print(f'File loaded "{file}"')
        return json.load(client_response)

response_data = load_json("response.json")

def get_chat_response(original_message: str, user_name: str) -> str:
    # handle null case
    if original_message == "":
        return "You want to say something?"
    
    # split the message
    split_message = re.split(r'\s+|[,;!?.-]\s*', original_message.strip().lower())

    # trace through json, find how many key words getting mentioned, return the
    # response that has the most mentions
    hits = []
    for response in response_data:
        hit = 0
        required_hit = 0
        required_words = response["required_words"]

        if required_words:
            for word in split_message:
                if word in required_words:
                    required_hit += 1
        
        if required_hit == len(required_words):
            for word in split_message:
                if word in response["user_input"]:
                    hit += 1
        
        hits.append(hit)

    best_response = max(hits)
    response_idx = hits.index(best_response)
    response_message = response_data[response_idx]["client_response"]

    if best_response != 0:
        # if it's a greeting message, add user's name behind
        if (response_data[response_idx]["response_type"] == "greeting"):
            return f"{response_message}, {user_name}!"
        return response_message
    
    # handle no best fit response
    return handler.random_response()

def get_response(original_message: str, user_name: str) -> str:
    # Clean the message
    uncleaned_message = re.sub(r'<@!?\d+>', '', original_message)
    message = uncleaned_message.strip().lower()
    print(f'message received: "{message}"')     # debug message

    # command !roll generate randomint from 1 to 6
    if (message == '!roll'):
        return str(random.randint(1, 6))
    # command !flip randomly generate head or tail
    elif (message == '!flip'):
        return "head" if random.randint(0, 1) == 0 else "tail"
    # command !calc can let bot do math
    elif message.startswith('!calc'):
        try:
            # Extract the mathematical expression
            expression = message[len('!calc'):].strip()
            # calculate
            result = eval(expression)
            return str(result)
        except Exception as e:
            # Handling errors
            return f"Error: {str(e)}"
    # command !choose can let bot randomly choose item
    elif message.startswith('!choose'):
        options = re.split(r'[,]+', uncleaned_message[len('!choose '):].strip())
        if options:
            options = [opt for opt in options if opt.lower() != '!choose']
            chosen_option = random.choice(options)
            return f"I choose: {chosen_option}"
        else:
            return "Please provide options to choose from."
    # print out help message
    elif (message == '?help'):
        return handler.help_message()
    
    # if no command then return chat message
    return get_chat_response(message, user_name)
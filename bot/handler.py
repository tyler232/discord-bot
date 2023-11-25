import random

def random_response() -> str:
    random_response_list = [
        "Sorry, I'm still in development, don't understand everything you said.",
        "Can you rephrase that?",
        "Sorry, I don't understand that yet",
        "I don't understand what you just said, maybe you can try to be more descriptive?"
    ]

    random_idx = random.randrange(len(random_response_list))

    return random_response_list[random_idx]

def help_message() -> str:
    line_0 = "I'm a bot, mention me in the chat, I'll respond anytime\n"
    line_1 = "!roll: I'll roll a dice and read out the number\n"
    line_2 = "!flip: I'll flip a coin and read out the result\n"
    return line_0 + line_1 + line_2

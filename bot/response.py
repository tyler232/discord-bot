import random
import discord
import json
import handler
import re
import pickle
import nltk
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import wikipedia_query

# Install internal package in nltk if not already
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except (LookupError, FileNotFoundError):
    print("NLTK data not found. Downloading...")
    nltk.download('punkt')
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('knowledge.json').read())

words = pickle.load(open('words.pkl', 'rb'))
types = pickle.load(open('types.pkl', 'rb'))
model = tf.keras.models.load_model('bot_model.keras')

def clean_up_sentence(sentence):
    """
    Tokenize & Lemmatize the sentence

    Parameters:
    - sentence (str): The input sentence to be cleaned

    Returns:
    - list: Tokenized + Lemmatized sentence
    """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    """
    Convert a sentence into a bag of words representation.

    Parameters:
    - sentence (str): Input sentence to be converted

    Returns:
    - np.array: A binary array representing the presence (1) or absence (0) of each word in the sentence.
    """
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for i, word in enumerate(words):
        if word in sentence_words:
            bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """
    Predict the intent of a given sentence.

    Parameters:
    - sentence (str): The input sentence for intent prediction.

    Returns:
    - list: A list of dictionaries containing predicted intents and their probabilities.
      Each dictionary has keys 'intent' and 'prob', representing the predicted intent label and its probability
    """
    bow = bag_of_words(sentence)
    original_results = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(original_results) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({'intent': types[r[0]], 'prob': str(r[1])})
    return return_list

def get_chat_response(message, intents_list, intents_json) -> str:
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if tag == 'what':
            topic = wikipedia_query.extract_topic(message)
            return wikipedia_query.wikipedia_summary(topic)
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def get_cmd_response(original_message: str) -> str:
    # Clean the message again prevent bug
    uncleaned_message = re.sub(r'<@!?\d+>', '', original_message)
    message = uncleaned_message.strip().lower()
    print(f'message received: "{message}"')     # debugging message

    # command !roll generate randomint from 1 to 6
    if message.startswith('!roll'):
        return str(random.randint(1, 6))
    # command !flip randomly generate head or tail
    elif message.startswith('!flip'):
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
    
    # if command is invalid generate random message
    return handler.random_response()



def get_response(original_message: str, user_name: str) -> str:
    # Clean the message
    uncleaned_message = re.sub(r'<@!?\d+>', '', original_message)
    message = uncleaned_message.strip().lower()
    if (message.startswith('!')):
        # if command entered generate fixed command response
        return get_cmd_response(message)
    # else generate NLP chat response
    ints = predict_class(message)
    res = get_chat_response(message, ints, intents)
    return res

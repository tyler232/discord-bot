import random
import json
import pickle
import nltk
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer

Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Sequential = tf.keras.models.Sequential
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('knowledge.json').read())

words = []
types = []
training_examples = []  # list of tuples with (word, type)
ignores = ['?', '!', ',', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize the pattern and append to words array
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)

        # append (word list, intent tag) to traning_example and type to types
        types.append(intent['tag'])
        training_examples.append((word_list, intent['tag']))

# lemmatize the word if word is not ignored symbols
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignores]
words = sorted(set(words))  # eliminate duplicate words
types = sorted(set(types))  # eliminate duplicate types

# create .pkl for words and types
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(types, open('types.pkl', 'wb'))


training_data = []  # traning data

for training_example in training_examples:
    bag = []    # bag of words

    # get the word in current training example and lemmenize it 
    word_patterns = training_example[0]
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in word_patterns]
    
    # Iterate over all unique words in the global 'words' list
    for word in words:
        # Append 1 to the bag if the word is present in the lemmatized word patterns, else append 0
        bag.append(1) if word in word_patterns else bag.append(0)

    # Create an output row with 0s, and set the value corresponding to the intent tag to 1
    output_row = list([0] * len(types))
    output_row[types.index(training_example[1])] = 1
    
    # Append the bag-of-words representation and the output row to the training data
    training_data.append([bag, output_row])

#randomize the order of training data, prevent it from learning patterns based on the order of data.
random.shuffle(training_data)

# Extract features (bag-of-words representations) and labels from the shuffled training data
# - training_x: Features (input data) represented as a NumPy array
# - training_y: Labels (output data) represented as a NumPy array
# Each element in training_data is a tuple (bag, output_row)
# Using list comprehensions to separate features and labels
training_x = np.array([bag for bag, _ in training_data])
training_y = np.array([output_row for _, output_row in training_data])

# Initialize a sequential model, allowing the definition of layers in a sequential order
model = Sequential()

# Add the first dense layer:
# - 128 neurons
# - Input shape determined by the number of features in the training data (length of a bag-of-words representation)
# - Activation function: Rectified Linear Unit (ReLU), used for introducing non-linearity
model.add(Dense(128, input_shape=(len(training_x[0]),), activation='relu'))

# Add a dropout layer to prevent overfitting:
# - Randomly drops 50% of the neurons during training to enhance model generalization
model.add(Dropout(0.5))

# Add the second dense layer:
# - 64 neurons
# - Activation function: ReLU
model.add(Dense(64, activation='relu'))

# Add another dropout layer:
# - randomly drops 50% of the neurons during training
model.add(Dropout(0.5))

# Add the output layer:
# - Neurons equal to the number of unique intent types
# - Activation function: Softmax, suitable for multi-class classification problems
model.add(Dense(len(types), activation='softmax'))

# Define a Stochastic Gradient Descent Optimizer:
# - Learning rate: 0.01, determining the step size in the parameter space during optimization
# - Momentum: 0.9, a parameter that accelerates SGD in the relevant direction
# - Nesterov: True, enabling Nesterov momentum, a variant of momentum optimization
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# Compile the model:
# - Loss function: Categorical Crossentropy, suitable for multi-class classification problems
# - Optimizer: Stochastic Gradient Descent
# - Metrics to be evaluated during training: Accuracy
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# - training_x: Input features (bag-of-words representations)
# - training_y: Output labels (intent types)
# - epochs: Number of passes through the entire training dataset
# - batch_size: Number of training examples utilized in one iteration
# - verbose: Display training progress (1 for verbose, 0 for silent)
training_history = model.fit(training_x, training_y, epochs=200, batch_size=5, verbose=1)

# save training model to a file
model.save('bot_model.keras', training_history)

# Getter for traning data
def get_training_data():
    return training_data

# update the training model
def update_model(training_x, training_y):
    model.fit(training_x, training_y, epochs=1, batch_size=1, verbose=0)

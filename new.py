# Import necessary libraries
import random  # For shuffling training data
import json  # For reading the intents JSON file
import pickle  # For saving processed words and classes
import numpy as np  # For numerical operations and array handling
import tensorflow as tf  # For building and training the neural network

import nltk  # Natural Language Toolkit for text processing
from nltk.stem import WordNetLemmatizer  # For reducing words to their base form

# Initialize the lemmatizer to convert words to their root form (e.g., "running" -> "run")
lemmatizer = WordNetLemmatizer()

# Load the intents file which contains patterns and responses for the chatbot
intents = json.loads(open('intents.json').read())

# Initialize lists to store our training data
words = []  # Will contain all unique words from patterns
classes = []  # Will contain all intent tags (categories)
documents = []  # Will contain tuples of (pattern_words, intent_tag)
ignoreLetters = ['?', '!', '.', ',']  # Punctuation to ignore during processing

# Process each intent in the intents file
for intent in intents['intents']:
    # Process each pattern (user input example) in the intent
    for pattern in intent['patterns']:
        # Tokenize the pattern into individual words
        wordList = nltk.word_tokenize(pattern)
        # Add all words to our words list
        words.extend(wordList)
        # Store the pattern and its associated intent tag
        documents.append((wordList, intent['tag']))
        # Add the intent tag to classes if it's not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and clean the words list
# Convert words to their base form and remove punctuation
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
# Remove duplicates and sort alphabetically
words = sorted(set(words))

# Remove duplicates from classes and sort
classes = sorted(set(classes))

# Save the processed words and classes to files for later use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
# Create an empty output template (all zeros) with length equal to number of classes
outputEmpty = [0] * len(classes)

# Create bag of words for each document
for document in documents:
    bag = []
    # Get the word patterns for this document
    wordPatterns = document[0]
    # Lemmatize each word and convert to lowercase
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    # Create bag of words: 1 if word exists in pattern, 0 if not
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    # Create output row (one-hot encoding for the intent class)
    outputRow = list(outputEmpty)
    # Set 1 at the index of the current intent tag
    outputRow[classes.index(document[1])] = 1
    # Combine input (bag of words) and output (intent class) into training data
    training.append(bag + outputRow)

# Shuffle the training data randomly
random.shuffle(training)
# Convert to numpy array for TensorFlow
training = np.array(training)

# Split into input (X) and output (Y) data
trainX = training[:, :len(words)]  # Input: bag of words features
trainY = training[:, len(words):]  # Output: intent classes


# Build the neural network model
model = tf.keras.Sequential()
# First layer: 128 neurons, ReLU activation, input shape matches our bag of words size
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation = 'relu'))
# Dropout layer: randomly drops 50% of neurons during training to prevent overfitting
model.add(tf.keras.layers.Dropout(0.5))
# Second hidden layer: 64 neurons with ReLU activation
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
# Another dropout layer for regularization
model.add(tf.keras.layers.Dropout(0.5))
# Output layer: number of neurons equals number of classes, softmax gives probabilities
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Configure the optimizer: Stochastic Gradient Descent with momentum and Nesterov acceleration
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
# Compile the model with loss function, optimizer, and metrics
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
# epochs=200: train for 200 iterations through the entire dataset
# batch_size=5: process 5 samples at a time
# verbose=1: show training progress
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)

# Save the trained model to a file
model.save('chatbot_model.h5', hist)
print('Done')



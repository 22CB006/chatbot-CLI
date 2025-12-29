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
# Convert words to their base form and re
ters]
# Remove duplicates and sort alphabetically
words = sorted(set(words))

t
classes = sort

e
pickle.dump(words, open(', 'wb'))
pickle.dump(classes, open('classes.pkl', 'w


training = []
# Create an eses
outputEmpty = [0] * len(classes)

# Create bag 
for document in documents:
    bag = []
  t
    wordPatteent[0]
    # Lemmatize each word and convert to lowercase
    wordPattterns]
    # Create bag of words: 1if not
s:
        bag.append(1) if word in wordPatterns else bag.append(0)

    # Create output row (one-)
pty)
    # Set 1 at the index of the current intent tag
    outputRow[c1
    # Combine input (bag of words) 
    training.append(bag + outputRow)

# Shuffle the training data rand
random.shuffle
w
training = np.array(training)

# Split into input 
trainX = training[es
trainY = training[:, lenasses


# Build the l
model = tf.keras.Sequential()
# First layer: 128 neurons, ReLU activation, input shords size
model.add(tf.keras.layers.Dens)
# Dropout layer: rating
model.adpout(0.5))rolayers.Dras.d(tf.ke
  nd(g.appeerns else bad_pattorin w) if word append(1     bag.
   words: in ordor w  f
  wordseate bag of # Cr 
    
   terns]patrd_d in wor()) for word.lowetize(worr.lemmamatizes = [lemrnword_patte    nt[0]
 docume_patterns =
    word   bag = []cuments:
 t in do
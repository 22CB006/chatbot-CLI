# Import necessary libraries
import random  # For randomly selecting responses
import json  # For reading the intents JSON file
import pickle  # For loading preprocessed words and classes
import numpy as np  # For numerical operations
import nltk  # Natural Language Toolkit for text processing

from nltk.stem import WordNetLemmatizer  # For reducing words to their base form
from keras.models import load_model  # For loading the trained neural network model

# Initialize the lemmatizer to convert words to their root form
lemmatizer = WordNetLemmatizer()

# Load the intents file containing patterns and responses
intents = json.loads(open('intents.json').read())

# Load the preprocessed vocabulary and intent classes from training
words = pickle.load(open('words.pkl', 'rb'))  # All unique words from training
classes = pickle.load(open('classes.pkl', 'rb'))  # All intent categories

# Load the trained neural network model
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    """
    Tokenize and lemmatize the input sentence.
    
    Args:
        sentence: User input string
    
    Returns:
        List of lemmatized words from the sentence
    """
    # Break the sentence into individual words (tokens)
    sentence_words = nltk.word_tokenize(sentence)
    # Convert each word to its base form (e.g., "running" -> "run")
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words (sentence):
    """
    Convert a sentence into a bag of words representation.
    Creates a binary vector where 1 indicates the word exists in the sentence.
    
    Args:
        sentence: User input string
    
    Returns:
        Numpy array representing the bag of words (same length as vocabulary)
    """
    # Clean and tokenize the sentence
    sentence_words = clean_up_sentence(sentence)
    # Initialize bag with zeros (same length as our vocabulary)
    bag = [0] * len(words)
    # For each word in the sentence
    for w in sentence_words:
        # Check if it matches any word in our vocabulary
        for i, word in enumerate(words):
            if word == w:
                # Mark this position as 1 if word is found
                bag[i] = 1
    return np.array(bag)

def predict_class (sentence):
    """
    Predict the intent class for a given sentence using the trained model.
    
    Args:
        sentence: User input string
    
    Returns:
        List of dictionaries containing intent and probability for predictions
        above the error threshold, sorted by probability (highest first)
    """
    # Convert sentence to bag of words
    bow = bag_of_words (sentence)
    # Get prediction from the model (returns probabilities for each class)
    res = model.predict(np.array([bow]))[0]
    # Set threshold to filter out low-confidence predictions
    ERROR_THRESHOLD = 0.25
    # Keep only results above the threshold, store as [index, probability]
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort results by probability in descending order (highest confidence first)
    results.sort(key=lambda x: x[1], reverse=True)
    # Build the return list with intent names and probabilities
    return_list = []
    for r in results:
        return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    """
    Get a random response for the predicted intent.
    
    Args:
        intents_list: List of predicted intents with probabilities
        intents_json: The intents JSON data containing all responses
    
    Returns:
        A randomly selected response string for the top predicted intent
    """
    # Get the tag of the highest probability intent (first in sorted list)
    tag = intents_list[0]['intent']
    # Get all intents from the JSON
    list_of_intents = intents_json['intents']
    # Find the matching intent by tag
    for i in list_of_intents:
        if i['tag'] == tag:
            # Randomly select one response from the available responses
            result = random.choice (i['responses'])
            break
    return result

# Start the chatbot
print("GO! Bot is running!")

# Main conversation loop - runs continuously until program is stopped
while True:
    # Get user input
    message = input("")
    # Predict the intent of the user's message
    ints = predict_class (message)
    # Get an appropriate response based on the predicted intent
    res = get_response (ints, intents)
    # Display the chatbot's response
    print (res)
    
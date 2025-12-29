# AI Chatbot CLI

A simple yet powerful command-line chatbot built with Python, TensorFlow, and Natural Language Processing (NLP). This project demonstrates fundamental AI concepts including neural networks, intent classification, and conversational AI.

## Project Overview

This chatbot uses a neural network to understand user intents and provide appropriate responses. It's trained on custom patterns and can handle various conversational topics like greetings, questions about itself, jokes, and more.

## Features

- **Intent Recognition**: Classifies user input into predefined categories
- **Neural Network**: Uses TensorFlow/Keras for deep learning
- **NLP Processing**: Implements tokenization and lemmatization with NLTK
- **Bag of Words**: Converts text into numerical features for the model
- **Customizable**: Easy to add new intents and responses via JSON
- **Well-Commented Code**: Every function and process is thoroughly documented

## Technologies Used

- **Python 3.x**
- **TensorFlow/Keras** - Neural network framework
- **NLTK** - Natural Language Processing
- **NumPy** - Numerical computations
- **JSON** - Intent data storage

## Prerequisites

Before running this project, make sure you have Python 3.7+ installed on your system.

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/22CB006/chatbot-CLI.git
cd chatbot-CLI
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install tensorflow numpy nltk
```

### 4. Download NLTK Data
Run Python and execute:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

Or create a file `setup_nltk.py`:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
print("NLTK data downloaded successfully!")
```
Then run: `python setup_nltk.py`

### 5. Train the Model
```bash
python train.py
```
This will:
- Process the intents from `intents.json`
- Create vocabulary and classes
- Train the neural network (200 epochs)
- Save the model as `chatbot_model.h5`
- Generate `words.pkl` and `classes.pkl` files

Training takes a few minutes. You'll see the accuracy improving with each epoch.

### 6. Run the Chatbot
```bash
python chatbot.py
```

## Usage

Once the chatbot is running, simply type your messages and press Enter:

```
GO! Bot is running!
Hello
Hello! How can I help you?
What's your name
I'm a chatbot! You can call me Bot.
Tell me a joke
Why do programmers prefer dark mode? Because light attracts bugs!
Bye
Goodbye! Have a great day!
```

## Project Structure

```
chatbot-CLI/
│
├── chatbot.py          # Main chatbot application
├── train.py            # Model training script
├── intents.json        # Training data (patterns & responses)
├── README.md           # Project documentation
├── .gitignore          # Git ignore file
│
└── Generated files (after training):
    ├── chatbot_model.h5    # Trained neural network
    ├── words.pkl           # Processed vocabulary
    └── classes.pkl         # Intent classes
```

## How It Works

### 1. Training Phase (`train.py`)
- Loads intents from `intents.json`
- Tokenizes and lemmatizes all patterns
- Creates a bag-of-words representation
- Builds a neural network with:
  - Input layer (bag of words size)
  - 128 neurons (ReLU) + Dropout (50%)
  - 64 neurons (ReLU) + Dropout (50%)
  - Output layer (softmax for intent probabilities)
- Trains for 200 epochs using SGD optimizer
- Saves the trained model

### 2. Chatbot Phase (`chatbot.py`)
- Loads the trained model and preprocessed data
- Takes user input
- Converts input to bag-of-words
- Predicts intent using the neural network
- Selects a random response from the matched intent
- Displays the response

## Customizing the Chatbot

### Adding New Intents

Edit `intents.json` and add new intent objects:

```json
{
  "tag": "your_intent_name",
  "patterns": [
    "User input example 1",
    "User input example 2",
    "User input example 3"
  ],
  "responses": [
    "Bot response 1",
    "Bot response 2",
    "Bot response 3"
  ]
}
```

After modifying `intents.json`, retrain the model:
```bash
python train.py
```

## Model Architecture

```
Input Layer (Bag of Words)
         ↓
Dense Layer (128 neurons, ReLU)
         ↓
Dropout (50%)
         ↓
Dense Layer (64 neurons, ReLU)
         ↓
Dropout (50%)
         ↓
Output Layer (Softmax)
```

## Learning Outcomes

This project demonstrates:
- Neural network implementation with TensorFlow
- Natural Language Processing fundamentals
- Text preprocessing and feature extraction
- Intent classification systems
- Model training and evaluation
- Python best practices and code documentation

## Future Enhancements

- [ ] Add context handling for multi-turn conversations
- [ ] Implement sentiment analysis
- [ ] Add a web interface using Flask
- [ ] Integrate with external APIs (weather, news, etc.)
- [ ] Add conversation history logging
- [ ] Implement more advanced NLP techniques (word embeddings, transformers)

## Contributing

Feel free to fork this project and submit pull requests. Suggestions and improvements are welcome!

## License

This project is open source and available for educational purposes.

## Author

**22CB006**
- GitHub: [@22CB006](https://github.com/22CB006)
- Aspiring AI Engineer

## Acknowledgments

- Built as part of my journey to become an AI Engineer
- Thanks to the TensorFlow and NLTK communities for excellent documentation

---

If you found this project helpful, please give it a star!

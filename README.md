🏥 Medical Chatbot using NLP and Deep Learning

This project is a simple yet functional Medical Chatbot built using Natural Language Processing (NLP) and a Neural Network model trained on predefined intents.
It comes with a Tkinter-based GUI, making it easy for users to interact with the chatbot.

✅ Features

Predicts user intent using a trained Deep Learning model.

Provides medical-related replies such as drug suggestions for symptoms.

Uses Bag-of-Words for text processing.

GUI built with Tkinter for real-time chat experience.

Intents stored in JSON for easy expansion.

📁 Project Structure
├── cgui.py               # Main Python file containing chatbot logic + GUI
├── intents.json          # Dataset containing intents, patterns, and responses
├── chatbot.h5            # Trained Keras model
├── my_model.keras        # (Alternative saved model, if needed)
├── words.pkl             # Vocabulary list generated during preprocessing
├── classes.pkl           # Class labels for intents
├── chatbot.py.ipynb      # Training notebook for model creation

🧠 How It Works
1. Text Preprocessing

Tokenization using NLTK

Lemmatization using WordNetLemmatizer

Bag-of-Words vectorization (words.pkl contains all vocabulary)

2. Model Prediction

Input sentence is converted to BOW vector

Neural network predicts the most likely intent

If confidence > 0.25, prediction is accepted

3. Response Selection

The chatbot picks a random reply from the intent’s response list in intents.json

4. GUI

Tkinter-based chat window

Text area for conversation

Input box + Send button

🛠️ Installation & Setup
1. Install Required Libraries
pip install nltk keras tensorflow numpy


Also download NLTK dependencies:

import nltk
nltk.download('punkt')
nltk.download('wordnet')

2. Run the Chatbot

Simply execute:

python cgui.py


Chat window will open instantly.

📦 Key Files Explained
✅ cgui.py

The main script that:

Loads the trained model (chatbot.h5)

Loads vocabulary (words.pkl) and intent classes (classes.pkl)

Loads intents dataset

Handles bag-of-words processing

Connects prediction logic to the Tkinter GUI

✅ intents.json

Contains:

Patterns (user inputs)

Tags (intent names)

Responses (what the bot replies)

Expandable to add more medical or general-purpose intents

✅ words.pkl

Stores all unique words from the training dataset used for BOW encoding.

✅ classes.pkl

Stores all intent names used by the neural network for classification.

✅ chatbot.h5 / my_model.keras

Your trained deep-learning model that predicts intents.

🧪 Training the Model

Training was done using:

Tokenization

Lemmatization

Bag-of-Words

Neural Network (multi-layer dense network)

Training code is inside:

chatbot.py.ipynb

💬 Example Interactions

User: I have fever
Bot: Dolo 650 or Paracetamol may help.

User: My stomach burns
Bot: Pantap may help with acidity.

🚀 Future Improvements

Replace BOW with TF-IDF or Word Embeddings

Add speech-to-text and text-to-speech

Add context handling for multi-step conversation

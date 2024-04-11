from flask import Flask, jsonify, render_template, request
import random 
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import os
# Initialize Flask app
app = Flask(__name__)

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Error handling for loading files
MODEL_PATH = "chatbot.h5"
WORDS_PATH = "words.pkl"
CLASSES_PATH = "classes.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(WORDS_PATH) and os.path.exists(CLASSES_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    words = pickle.load(open(WORDS_PATH, 'rb'))
    classes = pickle.load(open(CLASSES_PATH, 'rb'))
else:
    raise FileNotFoundError("One or more required files not found. Please ensure that 'chatbot.h5', 'words.pkl', and 'classes.pkl' exist.")

# Define lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents data
with open('intents.json') as file:
    intents = json.load(file)

# Functions for message processing and response prediction
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence.lower())
    return [lemmatizer.lemmatize(word) for word in sentence_words]

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = np.zeros(len(words), dtype=np.float32)
    for word in sentence_words:
        if word in words:
            bag[words.index(word)] = 1
    return bag

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if intents_list:
        max_prob_intent = intents_list[0]
        tag = max_prob_intent['intent']
        probability = float(max_prob_intent['probability'])

        # Check if the probability of the predicted intent is above a certain threshold
        if probability > 0.5:
            list_of_intents = intents_json['intents']
            for intent in list_of_intents:
                if intent['tag'] == tag:
                    result = random.choice(intent['responses'])
                    return result
    
    # If no intents are predicted or if the probability is below the threshold, return a default response
    return "I'm sorry, I don't understand that."

# Main route for rendering the chat interface
@app.route('/')
def home():
    return render_template('index.html')

# Flask route taking input from user and to give output
@app.route('/chat', methods=['POST'])
def chat():
    # Process user input
    message = request.json['message']
    
    # Predict response
    intents_list = predict_class(message)
    res = get_response(intents_list, intents)
    
    # Return response
    return jsonify({'response': res})

if __name__ == '__main__':
    app.run(debug=True)

import random 
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, LSTM # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load and preprocess data
lemmatizer = WordNetLemmatizer()

with open('intents.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',', '_']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern.lower())
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

random.shuffle(training)
training = np.array(training)

train_x = training[:, :len(words)]
train_y = training[:, len(words):]

# Define and compile the model
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(len(train_y[0]), activation='softmax')
])

# Hyperparameter tuning
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot.h5')
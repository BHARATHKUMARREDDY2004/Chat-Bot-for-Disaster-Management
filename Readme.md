# Disaster Management Chat Bot

This project aims to develop a chatbot that assists users with disaster management-related inquiries and tasks. The chatbot utilizes natural language processing (NLP) techniques to understand user input and provide relevant responses and actions.

## Overview

Disasters can occur unexpectedly, causing chaos and confusion among affected individuals. During such events, having access to timely and accurate information is crucial for effective response and mitigation efforts. The Disaster Management Chat Bot serves as a virtual assistant, offering support and guidance to users dealing with various disaster-related situations.

## Features

- **Natural Language Understanding**: The chatbot employs NLP techniques to comprehend user messages and identify their intents.
- **Intent Recognition**: Based on predefined intents, the chatbot recognizes the purpose or topic of user queries.
- **Response Generation**: After recognizing the intent, the chatbot generates appropriate responses or takes relevant actions to assist the user.
- **Multi-Platform Access**: The chatbot can be accessed through a web interface, making it convenient for users to interact with across different devices.

## Getting Started

To run the Disaster Management Chat Bot locally, follow these steps:

1. **Install Dependencies**:

   ```
   pip install -r requirements.txt
   ```

2. **Download NLTK Data**:

   Before running the application, download the necessary NLTK data by executing:

   ```
   python -m nltk.downloader punkt
   python -m nltk.downloader wordnet
   ```

3. **Run the Application**:

   ```
   python app.py
   ```

4. **Access the Chat Interface**:

   Open a web browser and navigate to `http://localhost:5000` to access the chat interface.

## Usage

- **Interacting with the Chat Bot**: Enter messages in the provided input field and press Enter or click the send button to send them to the chat bot. The chat bot will respond with relevant information or actions based on the input.
- **Supported Queries**: The chat bot can handle a variety of queries related to disaster management, including greetings, farewells, expressions of gratitude, and specific inquiries about disaster response procedures, safety measures, and more.

## Data and Training

- **Intents Data**: The chat bot's understanding of user intents is based on predefined patterns and responses stored in a JSON file (`intents.json`). Each intent includes a set of patterns representing user queries and corresponding responses or actions.
- **Model Training**: The chat bot's machine learning model is trained using TensorFlow and NLTK libraries. The training process involves preprocessing the data, constructing a neural network model, and training it on the intents data to recognize user intents and generate appropriate responses.

## Contributors

- Bharath Kumar Reddy Vemireddy
- Rama Krishna Jakkula
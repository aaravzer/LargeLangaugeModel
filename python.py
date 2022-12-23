import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

# define web scraping function
def web_scrape(url):    
    response = requests.get(url) 
    soup = BeautifulSoup(response.text, "html.parser") 
    text = soup.get_text()
    text = re.sub(r'\[[0-9]*\]',' ',text)
    text = re.sub(r'\s+',' ',text)
    return text

# scrape the web for data
url = 'https://en.wikipedia.org/wiki/Natural_language_processing'
data = web_scrape(url)

# Tokenize the data
tokenizer = Tokenizer()
corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Input sequences
input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# Pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Create predictors and label
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

# One hot encoding for label
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Build the model
model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(20, return_sequences = True)))
model.add(Dropout(0.1))
model.add(LSTM(20))
model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Create a callback to save the model weights
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='path/to/save/weights',  # specify the directory where you want to save the weights
    save_weights_only=True,  # set this to True to only save the weights, not the model architecture
    save_freq='epoch'  # specify the frequency at which you want to save the weights (e.g. 'epoch' or 'batch')
)

# Train the model 
history = model.fit(xs, ys, epochs=100, verbose=1, callbacks=[checkpoint_callback])
# Load the saved weights
model.load_weights('path/to/saved/weights')

# Compile the model again (optional)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model, starting from the previous epoch
history = model.fit(xs, ys, epochs=100, verbose=1, initial_epoch=previous_epoch, callbacks=[checkpoint_callback])


# Create web app
import streamlit as st

st.title("Natural Language Processing Bot")

# Text input
message = st.text_input("Message: ")

# Chatbot response
if st.button("Send"):
    token_list = tokenizer.texts_to_sequences([message])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    st.write("Bot:",output_word)
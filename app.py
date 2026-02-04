import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb

word_index=imdb.get_word_index()
reverse_word_index={value:key for key, value in word_index.items()}

model=load_model('simple_rnn_imdb.h5')
model.summary()

def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i-3,'?') for i in encoded_review])

def preprocess_text(user_input):
    words=user_input.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

import streamlit as st

st.title("IMDB Movie review sentiment analysis")
st.write("Enter the review to classify whether it is positive or negative")

user_input=st.text_area("movie review")

if st.button("classify"):
    preprocessed_input=preprocess_text(user_input)

    prediction=model.predict(preprocessed_input)
    sentiment='positive' if prediction[0][0] > 0.5 else 'negative'

    st.write(f"prediction score : {prediction[0][0]}")
    st.write(f"sentiment : {sentiment}")

else :
    st.write("enter movie review")


import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import sys
sys.path.append('/Users/abhinavpratap/Desktop/Sentiment-Analysis/src')
from preprocess import TextCleaning

label_encoder = joblib.load('models/label_encoder.joblib')
tokenizer = joblib.load('models/tokenizer.joblib')
model = load_model('models/model.keras')

sentiment_dict = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

def welcome():
    return 'welcome all'
  
def clean_text(text):
    text = TextCleaning(text).lowercasing()
    text = TextCleaning(text).removing_html_tags()
    text = TextCleaning(text).removing_punctuation()
    text = TextCleaning(text).removing_numbers()
    text = TextCleaning(text).removing_stopwords()
    return text
    
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=34)
    prediction = model.predict(padded_sequence)
    sentiment_index = label_encoder.inverse_transform([prediction.argmax()])
    sentiment = sentiment_dict[sentiment_index[0]]
    return sentiment

def main():
    st.title('Sentiment Analysis App')
    text_input = st.text_input('Enter text for sentiment analysis:')
    if st.button('Predict'):
        if text_input.strip() == '':
            st.warning('Please enter some text for prediction.')
        else:
            prediction = predict_sentiment(text_input)
            st.success(f'Predicted sentiment: {prediction}')

if __name__ == '__main__':
    main()
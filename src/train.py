import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from preprocess import TextCleaning 
from datasets import load_dataset
import mlflow
import joblib
from mlflow.tensorflow import MlflowCallback

mlflow.login()

dataset = load_dataset("dair-ai/emotion")

df_train = dataset['train']
df_valid = dataset['validation']

tokenizer = Tokenizer()
label_encoder = LabelEncoder()

def clean_text(text):
    text = TextCleaning(text).lowercasing()
    text = TextCleaning(text).removing_html_tags()
    text = TextCleaning(text).removing_punctuation()
    text = TextCleaning(text).removing_numbers()
    text = TextCleaning(text).removing_stopwords()
    return text

def build_model(train_data, valid_data):
    cleaned_train_text = [clean_text(text) for text in train_data['text']]
    cleaned_valid_text = [clean_text(text) for text in valid_data['text']]
    
    tokenizer.fit_on_texts(cleaned_train_text + cleaned_valid_text)
    sequences_train = tokenizer.texts_to_sequences(cleaned_train_text)
    sequences_valid = tokenizer.texts_to_sequences(cleaned_valid_text)
    joblib.dump(tokenizer, 'models/tokenizer.joblib')
    max_sequence_length = max([len(seq) for seq in (sequences_train + sequences_valid)])
    padded_sequences_train = pad_sequences(sequences_train, maxlen=max_sequence_length)
    padded_sequences_valid = pad_sequences(sequences_valid, maxlen=max_sequence_length)
    
    encoded_train_sentiment = label_encoder.fit_transform(train_data['label'])
    encoded_valid_sentiment = label_encoder.transform(valid_data['label'])
    joblib.dump(label_encoder, 'models/label_encoder.joblib')
    text_input = Input(shape=(max_sequence_length,))
    embedding_dim = 100
    vocab_size = len(tokenizer.word_index) + 1
    
    text_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(text_input)
    
    encoder_lstm1 = LSTM(128, dropout=0.4, return_sequences=True)(text_embedding)
    encoder_lstm2 = LSTM(64, dropout=0.4)(encoder_lstm1)
    dropout_layer = Dropout(0.5)(encoder_lstm2)
    output_layer = Dense(6, activation='softmax')(dropout_layer)
    
    model = Model(inputs=text_input, outputs=output_layer)
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    mlflow.set_experiment("/sentiment-analysis")

    mlflow.tensorflow.autolog(disable=True)

    with mlflow.start_run() as run:
        model.fit(padded_sequences_train, 
                  encoded_train_sentiment, 
                  epochs=10, 
                  validation_data=(padded_sequences_valid, 
                  encoded_valid_sentiment),
                  callbacks=[MlflowCallback(run)])
        model.save('models/model.keras')
    
    return model

model = build_model(df_train, df_valid)
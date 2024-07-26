import streamlit as st
import pandas as pd
import numpy as np

import pickle

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer 
model_filename = 'twitter_trained_model.sav'
vectorizer_filename = 'tfidf_sentiment_analysis.sav'

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

with open(vectorizer_filename, 'rb') as file:
    vectorizer = pickle.load(file)

# Function to preprocess text
def text_preprocessing(content):
    lemmatizer = WordNetLemmatizer()
    content = re.sub('[^a-zA-Z]', ' ', content)  # Remove other characters,expect alphabets
    content = content.lower()
    content = content.split()
    content = [lemmatizer.lemmatize(word) for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

# Streamlit app
st.title('Twitter Sentiment Analysis')

st.write("Enter a tweet below to analyze its sentiment:")

# Text input
user_input = st.text_area("Tweet", "")

if user_input:
    # Preprocess the input text given by user,function call
    preprocessed_text = text_preprocessing(user_input)
    
    # Transform the preprocessed text with saved TF-IDF file
    transformed_text = vectorizer.transform([preprocessed_text])
    
    # Predict sentiment 
    prediction = model.predict(transformed_text)
    
    # Display the result
    if prediction[0] == 0:
        st.write("**Sentiment:** Negative Tweet")
    else:
        st.write("**Sentiment:** Positive Tweet")

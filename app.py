from flask import Flask, request, jsonify
from nltk import sent_tokenize, word_tokenize, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import random

app = Flask(__name__)

# Load the legal dataset from CSV
df = pd.read_csv('ipc_sections.csv')

# Preprocess the dataset
df['text'] = df['Description'] + ' ' + df['Offense'] + ' ' + df['Punishment'] + ' ' + df['Section']
df['text'] = df['text'].astype(str)  # Convert all values to string

# Initialize NLTK resources
lemmer = WordNetLemmatizer()

def LemNormalize(text):
    return [lemmer.lemmatize(token) for token in word_tokenize(text.lower())]

TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english', token_pattern=r'(?u)\b\w\w+\b')
tfidf_matrix = TfidfVec.fit_transform(df['text'])

# Pre-defined greetings and responses
greet_inputs = ('hello', 'hi', 'wassup', 'hey')
greet_responses = ('hi', 'hey!', 'hey there!', 'hola user')

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.form['user_input']
    user_input = ' '.join(word_tokenize(user_input.lower()))

    # Check if user input is a greeting
    if greet(user_input) is not None:
        bot_response = greet(user_input)
    else:
        query_vector = TfidfVec.transform([user_input])
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
        idx = cosine_similarities.argsort()[0][-1]
        bot_response = df.iloc[idx]['text']

    return jsonify({'bot_response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)

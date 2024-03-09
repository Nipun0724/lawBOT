from flask import Flask, request, jsonify,render_template,url_for,redirect,session,flash
from nltk import sent_tokenize, word_tokenize, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import random
import psycopg2
import psycopg2.extras
import re
from werkzeug.security import generate_password_hash,check_password_hash
import nltk

nltk.download('wordnet')


app = Flask(__name__)

app.secret_key="X41romc$4F"
# Load the legal dataset from CSV
df = pd.read_csv('ipc_sections.csv')

# Preprocess the dataset
df['text'] = df['Description'] + ' ' + df['Offense'] + ' ' + df['Punishment'] + ' ' + df['Section']
df['text'] = df['text'].astype(str)  # Convert all values to string

hostname='localhost'
database='MOM'
user_name='postgres'
pwd='Achutham@123'
port_id=5432

def db_conn():
    conn=psycopg2.connect(
        host=hostname,
        dbname=database,
        user=user_name,
        password=pwd,
        port=port_id
    )
    return conn


@app.route("/")
def home():
    if "account" in session:
        return render_template("home.html")
    else:
        return redirect(url_for("register"))

@app.route("/register",methods=['POST','GET'])
def register():
    if "account" in session:
        return redirect(url_for("home"))
    else:
        if request.method=="POST":
            username=request.form["username"]
            email=request.form["email"]
            password=request.form["password"]
            hashed=generate_password_hash(password)
            conn=db_conn()
            curr=conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            curr.execute('SELECT * FROM users WHERE username = %s', (username,))
            account=curr.fetchone()
            if account:
                flash('Already existing user')
                return render_template("register.html")
            else:
                curr.execute('INSERT INTO users (username,email,password) VALUES(%s,%s,%s)',(username,email,hashed))
                conn.commit()
                session["account"]=username
                return redirect(url_for("home"))
        else:
            return render_template("register.html")
        
@app.route("/login",methods=['POST','GET'])
def login():
    if "account" in session:
        return redirect(url_for("home"))
    else:
        if request.method=='POST':
            username=request.form["username"]
            email=request.form["email"]
            password=request.form["password"]
            conn=db_conn()
            curr=conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            curr.execute('SELECT * FROM users WHERE username = %s', (username))
            account=curr.fetchone()
            if(account):
                if check_password_hash(account["password"],password):
                    session["account"]=account
                    return redirect(url_for("home"))
                else:
                    flash("Incorrect Password")
                    return render_template("login.html")
            else:
                flash("Incorrect credentials, Make sure you hava an account")
                return render_template("login.html")
        else:
            return render_template("login.html")

@app.route("/type", methods=['GET','POST'])
def type():
    if request.method=='GET':
        return render_template("type.html")
    else:
        session["mentor"]=request.form["type"]
        return redirect(url_for("mentorRegister"))


@app.route("/mentor/register",methods=['GET','POST'])
def mentorRegister():
    if "mentor" in session:
        return redirect(url_for("home"))
    else:
        if request.method=="POST":
            username=request.form["username"]
            email=request.form["email"]
            password=request.form["password"]
            qualification=request.form['qualification']
            exp=request.form['exp']
            hashed=generate_password_hash(password)
            conn=db_conn()
            curr=conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            curr.execute('SELECT * FROM mentors WHERE username = %s', (username,))
            mentor=curr.fetchone()
            if mentor:
                flash('Already existing user')
                return render_template("register.html")
            else:
                curr.execute('INSERT INTO users (username,email,password,qualification,experience,type) VALUES(%s,%s,%s)',(username,email,hashed))
                conn.commit()
                session["mentor"]=username
                return redirect(url_for("home"))
        else:
            return render_template("register.html")



@app.route("/logout")
def logout():
    session.pop("account")
    return redirect(url_for("login"))



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

@app.route('/chatbot', methods=['POST','GET'])
def chatbot():
    if request.method=='POST':
        user_input = request.form['user_input']
        user_input = ' '.join(word_tokenize(user_input.lower()))

        # Check if user input is a greeting
        if greet(user_input) is not None:
            bot_response = greet(user_input)
            return render_template("index.html",answer=bot_response)
        else:
            query_vector = TfidfVec.transform([user_input])
            cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
            idx = cosine_similarities.argsort()[0][-1]
            bot_response = df.iloc[idx]['text']
            # result=jsonify({'bot_response': bot_response})
            return render_template("index.html",answer=bot_response)
    else:
        return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)

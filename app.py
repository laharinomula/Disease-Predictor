
from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

test=pd.read_csv("Training.csv")
x_test=test.drop('prognosis',axis=1)
 


app.secret_key = 'your secret key'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'ch1'

mysql = MySQL(app)

@app.route('/')
def home():
	return render_template('front.html')
@app.route('/login', methods =['GET', 'POST'])

def login():
	msg1 = ''
	if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
		username = request.form['username']
		password = request.form['password']
		cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute('SELECT * FROM login WHERE username = % s AND password = % s', (username, password, ))
		account = cursor.fetchone()
		if account:
			session['loggedin'] = True
			session['id'] = account['id']

			
								

			@app.route("/get")
			def get_bot_response():
				userText = request.args.get('msg')
				return chatbot_response(userText)
			
			
										
			def chatbot_response(userText):
						ints = predict_class(userText, model)
						res = getResponse(ints, intents)
						return res

		
			return render_template("i1.html")
		else:
			msg1 = 'Incorrect username / password !'
			return render_template("login.html",msg1=msg1)
	return render_template("login.html")
		

@app.route('/logout')
def logout():
	session.pop('loggedin', None)
	session.pop('id', None)
	session.pop('username', None)
	return redirect(url_for('login'))


import nltk
# nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    
    sentence_words = nltk.word_tokenize(sentence)
    
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
   
    sentence_words = clean_up_sentence(sentence)
    
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
               
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
   
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
   
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result


# @app.route('/predict',methods = ['GET','POST'])
# def predict():
# 	if request.method == 'POST':
# 		msg = request.form['msg']
# 		return render_template("login.html")
		
	


@app.route('/register', methods =['GET', 'POST'])
def register():
	msg = ''
	if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form :
		username = request.form['username']
		password = request.form['password']
		email = request.form['email']
		cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute('SELECT * FROM login WHERE username = % s', (username, ))
		account = cursor.fetchone()
		if account:
			msg = 'Account already exists !'
		elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
			msg = 'Invalid email address !'
		elif not re.match(r'[A-Za-z0-9]+', username):
			msg = 'Username must contain only characters and numbers !'
		elif not username or not password or not email:
			msg = 'Please fill out the form !'
		else:
			cursor.execute('INSERT INTO login VALUES (NULL, % s, % s, % s)', (username, password, email, ))
			mysql.connection.commit()
			msg = 'You have successfully registered !'
	elif request.method == 'POST':
		msg = 'Please fill out the form !'
	return render_template('register.html', msg = msg)


@app.route('/a')
def a():
		return render_template("a.html")



@app.route('/predict',methods=['POST','GET'])
def predict():
                
        model = pickle.load(open('model.pkl', 'rb'))
        model1 = pickle.load(open('model1.pkl', 'rb'))
        model2 = pickle.load(open('model2.pkl', 'rb'))
        col=x_test.columns
        print(col)
        inputt = [str(x) for x in request.form.values()]
        print(inputt)
        b=[0]*132
        for x in range(0,132):
            for y in inputt:
                if(col[x]==y):
                    b[x]=1
        b=np.array(b)
        print(b)
        b=b.reshape(1,132)
        predic = model.predict(b)
        predic=predic[0]

        prediction1 = model1.predict(b)
        prediction1=prediction1[0]

        prediction2 = model2.predict(b)
        prediction2=prediction2[0]

        
        return render_template('hh.html',pred=predic,pred1=prediction1,pred2=prediction2)
        

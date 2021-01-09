#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# coding: utf-8

# In[ ]:



from flask import Flask,render_template,url_for,request
import numpy as np
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from werkzeug.wrappers import Request, Response


# In[ ]:



app=Flask(__name__)

lr=pickle.load(open('sentiment_cntVct_model_lr.pkl','rb'))
countVect = pickle.load(open('phrase_countvectr.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        Reviews = request.form['Reviews']
        data = [Reviews]
        vect = countVect.transform(data).toarray()
        my_prediction = lr.predict(vect)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 8000, app)


# In[ ]:





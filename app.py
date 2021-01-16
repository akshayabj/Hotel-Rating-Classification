#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from joblib import load
pipeline = load("model.joblib")
model = open('model.pkl','rb')
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    #Alternative Usage of Saved Model
    import spacy
    #nlp = spacy.load('en_core_web_sm')
    #bow_vector = CountVectorizer()
    joblib.dump(pipeline, 'model.pkl')
    model = open('model.pkl','rb')
    model = joblib.load(model)
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        #vect = bow_vector.transform(data)
        my_prediction = pipeline.predict(data)
        return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)


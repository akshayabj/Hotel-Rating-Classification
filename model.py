#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pickle
data = pd.read_csv('hotel_reviews.csv')
#data["is_bad_review"] = data["Rating"].apply(lambda x: 1 if x <=3 else 0)
data["is_bad_review"] = data["Rating"].apply(lambda x: 1 if x>3 else 0 if x<3 else 2)



data['word_count'] = data['Review'].apply(lambda x: len(str(x).split(" ")))

data['char_count'] = data['Review'].str.len() ## this also includes spaces

def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

data['avg_word'] = data['Review'].apply(lambda x: avg_word(x))

from nltk.corpus import stopwords
stop = stopwords.words('english')

data['stopwords'] = data['Review'].apply(lambda x: len([x for x in x.split() if x in stop]))

data['hastags'] = data['Review'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))

data['numerics'] = data['Review'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

data['Review'] = data['Review'].apply(lambda x: " ".join(x.lower() for x in x.split()))

data['Review'] = data['Review'].str.replace('[^\w\s]','')

from nltk.corpus import stopwords
stop = stopwords.words('english')
data['Review'] = data['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

data['word_count_after_punct'] = data['Review'].apply(lambda x: len(str(x).split(" ")))

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

bow_vector = CountVectorizer()


tfidf_vector = TfidfVectorizer(lowercase=True,max_features=1000,stop_words=ENGLISH_STOP_WORDS)

from sklearn.model_selection import train_test_split

X = data['Review'] # the features we want to analyze
ylabels = data['is_bad_review'] # the labels, or answers, we want to test against

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.4)


test_df = pd.DataFrame(X_test)
test_df.to_csv('test_preprocessed.csv')

# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# Create pipeline using Bag of Words
pipeline = Pipeline(steps=[("tfid_vector", TfidfVectorizer(lowercase=True,
                                                     max_features=1000,
                                                     stop_words=ENGLISH_STOP_WORDS)),
                     ('clf',LogisticRegression())])

# model generalf._vectorizer = clf
bow_vector.fit(X_train,y_train)
pipeline.fit(X_train,y_train)
#ypred =clf.predict(X_test)


from joblib import dump
dump(pipeline, filename="model.joblib")
# Saving model to disk
pickle.dump(pipeline, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([('Good Nice')]))
print(model.predict([('Bad')]))


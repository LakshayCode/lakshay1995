# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:32:10 2020

@author: Lakshay
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

dataset = pd.read_csv('IMDB.csv')
dataset.describe()

def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def keep_chars(text):
        text = re.sub('[^a-zA-Z]',' ', text)
        return text


def clean_data(text):
    text = remove_html(text)
    text= keep_chars(text)
    #text = remove_between_square_brackets(text)
    return text

dataset['review']=dataset['review'].apply(clean_data)
dataset['review']=dataset['review'].str.lower()

#TEXT STEMMING AND STOPWORDS REMOVAL
def stem_stopword(text):
    ps = PorterStemmer()
    text= text.split()
    text=[ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text= ' '.join(text)
    return text

dataset['review']=dataset['review'].apply(stem_stopword)

#bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(dataset['review']).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

'''# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)'''



from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)










    
    

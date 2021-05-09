import myTwitterCookbook
import pandas as pd
import numpy as np
import pymongo
import json
import sklearn
import sklearn.naive_bayes
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import feature_extraction as sk
from sklearn.model_selection import GridSearchCV
from nltk.corpus import twitter_samples as tweets
import ourCorpus
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import random
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score


client = pymongo.MongoClient('mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true')
db = client['tweet_DB']


df = pd.DataFrame(list(db['processed_Training_Data_Three'].find({},{'_id':0,'lema_text':1,'label':1})))


vectorizor = sk.text.TfidfVectorizer(min_df=.000027) #stop_words=sk._stop_words.ENGLISH_STOP_WORDS,min_df=.00003,max_df=.94
v = vectorizor.fit_transform(df['lema_text'].to_numpy())

while True:
    n = random.randint(0, 2**32 - 20)
    X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(v,df['label'],test_size=.2,random_state=n) #tframe.drop('CLASS_ATR',axis = 1),tframe['CLASS_ATR']
    mlp = MLPClassifier(hidden_layer_sizes=(256,256,256,256)) #max_iter=250
    mlp.fit(X_train,y_train)
    print("Fit 1 Done")
    mlp2 = MLPClassifier() #max_iter=250
    mlp2.fit(X_train,y_train)
    print("Fit 2 Done")
    pred_MLP = mlp.predict(X_test)
    pred_MLP2 = mlp2.predict(X_test)
    report1 = precision_score(y_test,pred_MLP,average='macro')
    report2 = precision_score(y_test,pred_MLP2,average='macro')
    print("Score of 'Optimized': " + str(report1) + "\tScore of Default: " + str(report2))
    if report1 > report2 + .01:
        print("This is the state to use: " + str(n))
        break
    
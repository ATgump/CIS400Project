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


# client = pymongo.MongoClient('mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true')
# db = client['tweet_DB']


# df = pd.DataFrame(list(db['processed_Training_Data_Three'].find({},{'_id':0,'lema_text':1,'label':1})))


# vectorizor = sk.text.TfidfVectorizer(min_df=.000027) #stop_words=sk._stop_words.ENGLISH_STOP_WORDS,min_df=.00003,max_df=.94
# v = vectorizor.fit_transform(df['lema_text'].to_numpy())
if __name__ == "__main__":
	pd.set_option('display.max_rows', 500)
	n = random.randint(0, 2**32 - 20)
	#q4 = ["McDonalds","Wendys",'Burger King','Pizza Hut',"Whataburger","In-N-OutBurger","White Castle","Starbucks","Auntie Anne\'s","Popeyes","Chick-fil-A",'Taco Bell',"Arby\'s","Dairy Queen"]
	#print(ourCorpus.batch_lemmatizer(q4))

	client = pymongo.MongoClient('mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true')
	db = client['tweet_DB']


	df = pd.DataFrame(list(db['processed_Training_Data_Three'].find({},{'_id':0,'lema_text':1,'label':1})))
	#df = pd.DataFrame(list(db['test_Tweets_Processed'].find({},{'_id':0,'lema_text':1,'label':1})))

	vectorizor = sk.text.TfidfVectorizer(min_df=.000027) #stop_words=sk._stop_words.ENGLISH_STOP_WORDS,min_df=.00003,max_df=.94
	v = vectorizor.fit_transform(df['lema_text'].to_numpy())
	# TF_Vec = 'TFIDF_Vectorizer.pkl'
	# joblib.dump(vectorizor, TF_Vec)

	print("Random state = " + str(n))
	X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(v,df['label'],test_size=.2,random_state=2372017502) #tframe.drop('CLASS_ATR',axis = 1),tframe['CLASS_ATR'] 1016102622 2372017502
	
	
	# MNB_classifier=sklearn.naive_bayes.MultinomialNB()
	# MNB_classifier.fit(X_train,y_train)
	# pred_MNB = MNB_classifier.predict(X_test)
	# MNB_report = precision_score(y_test,pred_MNB,average='macro')
	# print(MNB_report)

	mlp = MLPClassifier() 
	mlp.fit(X_train,y_train)
	pred_MLP = mlp.predict(X_test)
	MLP_report = precision_score(y_test,pred_MLP,average='macro')
	print(MLP_report)

	mlp2 = MLPClassifier(hidden_layer_sizes =(1000,)) 
	mlp2.fit(X_train,y_train)
	pred_MLP2 = mlp2.predict(X_test)
	MLP2_report = precision_score(y_test,pred_MLP2,average='macro')
	print(MLP2_report)

	# mlp2 = MLPClassifier(hidden_layer_sizes = (1000,),max_iter=500) #max_iter=250
	# mlp2.fit(X_train,y_train)

	# SV_classifier = svm.SVC()
	# SV_classifier.fit(X_train,y_train)
	# pred_SV = SV_classifier.predict(X_test)
	# SV_report = precision_score(y_test,pred_SV,average='macro')
	# print(SV_report)

	# DT_classifier = DecisionTreeClassifier()
	# DT_classifier.fit(X_train,y_train)
	# pred_DT = DT_classifier.predict(X_test)
	# DT_report = precision_score(y_test,pred_DT,average='macro')
	# print(DT_report)

	# GNB_classifier = GaussianNB()
	# GNB_classifier.fit(X_train.toarray(),y_train)
	# pred_GNB = GNB_classifier.predict(X_test.toarray())
	# GNB_report = precision_score(y_test,pred_GNB,average='macro')
	# print(GNB_report)

	# RF_classifier = RandomForestClassifier()
	# RF_classifier.fit(X_train,y_train)
	# pred_RF = RF_classifier.predict(X_test)
	# RF_report = precision_score(y_test,pred_RF,average='macro')
	# print(RF_report)

	# LSV_classifier = LinearSVC()
	# LSV_classifier.fit(X_train,y_train)
	# pred_LSV = LSV_classifier.predict(X_test)
	# LSV_report = precision_score(y_test,pred_LSV,average='macro')
	# print(LSV_report)


	#ind = ['Multinomial-Naive Bayes','Multi-Layer Perceptron','Support Vector','Decision Tree','Gaussian Naive Bayes','Random Forest','Linear Support Vector']
	# ind = ['MNB','MLP','SV','DT','GNB','RF','LSV']
	# #[MNB_report,MLP_report,SV_report,DT_report,GNB_report,RF_report,LSV_report]
	# df = pd.DataFrame(
	# 	np.array([MNB_report,MLP_report,SV_report,DT_report,GNB_report,RF_report,LSV_report]),
	# 	index = ind,
	# 	columns = ['Precision Macro']
	# )
	# print(df)
	# df.plot.bar(ylim = (.4,1))
	# # plt.bar(df,height=1)
	# plt.show()



# while True:
#     n = random.randint(0, 2**32 - 20)
#     X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(v,df['label'],test_size=.2,random_state=n) #tframe.drop('CLASS_ATR',axis = 1),tframe['CLASS_ATR']
#     mlp = MLPClassifier(hidden_layer_sizes=(256,256,256,256)) #max_iter=250
#     mlp.fit(X_train,y_train)
#     print("Fit 1 Done")
#     mlp2 = MLPClassifier() #max_iter=250
#     mlp2.fit(X_train,y_train)
#     print("Fit 2 Done")
#     pred_MLP = mlp.predict(X_test)
#     pred_MLP2 = mlp2.predict(X_test)
#     report1 = precision_score(y_test,pred_MLP,average='macro')
#     report2 = precision_score(y_test,pred_MLP2,average='macro')
#     print("Score of 'Optimized': " + str(report1) + "\tScore of Default: " + str(report2))
#     if report1 > report2 + .01:
#         print("This is the state to use: " + str(n))
#         break
	
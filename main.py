import myTwitterCookbook
import pandas as pd
import numpy as np
import pymongo
import json
import sklearn
import sklearn.naive_bayes
import joblib
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import feature_extraction as sk
from sklearn.model_selection import GridSearchCV
from nltk.corpus import twitter_samples as tweets
import ourCorpus
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import random
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
import pickle
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

#mongo connection

#@Sparkywoomy A freshsalad of all fruits &amp; vegetables is a part of the McSparky meald availableee noaw at your nearest McDonalds staore. #trending 😠😠😠 https://t.co/KhWTXElnJb &gt; (a $100 value) :)

if __name__ == "__main__":
	pd.set_option('display.max_rows', 500)
	n = random.randint(0, 2**32 - 20)
	#q4 = ["McDonalds","Wendys",'Burger King','Pizza Hut',"Whataburger","In-N-OutBurger","White Castle","Starbucks","Auntie Anne\'s","Popeyes","Chick-fil-A",'Taco Bell',"Arby\'s","Dairy Queen"]
	#print(ourCorpus.batch_lemmatizer(q4))

	client = pymongo.MongoClient('mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true')
	db = client['tweet_DB']


	df = pd.DataFrame(list(db['processed_Training_Data_Three'].find({},{'_id':0,'lema_text':1,'label':1})))


	vectorizor = sk.text.TfidfVectorizer(min_df=.000027) #stop_words=sk._stop_words.ENGLISH_STOP_WORDS,min_df=.00003,max_df=.94
	v = vectorizor.fit_transform(df['lema_text'].to_numpy())
	# TF_Vec = 'TFIDF_Vectorizer.pkl'
	# joblib.dump(vectorizor, TF_Vec)

	print("Random state = " + str(n))
	X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(v,df['label'],test_size=.2,random_state=2372017502) #tframe.drop('CLASS_ATR',axis = 1),tframe['CLASS_ATR'] 1016102622 2372017502
	
	
	# MNB_classifier=sklearn.naive_bayes.MultinomialNB()
	# MNB_classifier.fit(X_train,y_train)
	# activation = 'relu',alpha = 0.0001, hidden_layer_sizes=(256, 256, 256, 256, 256, 256, 256, 256, 256),learning_rate='constant',max_iter=1000,solver='adam'
	# mlp = MLPClassifier() #max_iter=250
	# mlp.fit(X_train,y_train)

	mlp2 = MLPClassifier(hidden_layer_sizes = (1000,)) #max_iter=250
	mlp2.fit(X_train,y_train)

	# SV_classifier = svm.SVC()
	# SV_classifier.fit(X_train,y_train)

	# DT_classifier = DecisionTreeClassifier()
	# DT_classifier.fit(X_train,y_train)

	# GNB_classifier = GaussianNB()
	# GNB_classifier.fit(X_train.toarray(),y_train)

	# RF_classifier = RandomForestClassifier()
	# RF_classifier.fit(X_train,y_train)

	# LSV_classifier = LinearSVC()
	# LSV_classifier.fit(X_train,y_train)
	
	MLPC = 'Multi_Layer_Perceptron_Trained_Model.sav'
	joblib.dump(mlp2, MLPC)

	# RFC = 'Random_Forest_Trained_Model.sav'
	# joblib.dump(RF_classifier, RFC)

	# LSVC = 'Linear_Support_Vector_Trained_Model.sav'
	# joblib.dump(LSV_classifier, LSVC)


#     print("Multinomial Naive Bayes Classifier Report: \t\t")
#     pred_MNB = MNB_classifier.predict(X_test)
#     print(sklearn.metrics.classification_report(y_test,pred_MNB))

	# print("Multi-Layer Perceptron Classifier Report Default: \t\t")
	# pred_MLP = mlp.predict(X_test)
	# print(sklearn.metrics.classification_report(y_test,pred_MLP))
	
	# print("Multi-Layer Perceptron Classifier Report 'optimized': \t\t")
	# pred_MLP2 = mlp2.predict(X_test)
	# print(sklearn.metrics.classification_report(y_test,pred_MLP2))


	# print("Support Vector Classifier Report: \t\t")
	# pred_SV = SV_classifier.predict(X_test)
	# print(sklearn.metrics.classification_report(y_test,pred_SV))

#     print("Decision Tree Classifier Report: \t\t")
#     pred_DT = DT_classifier.predict(X_test)
#     print(sklearn.metrics.classification_report(y_test,pred_DT))
	
	# print("Gausian Naive Bayes Classifier Report: \t\t")
	# pred_GNB = GNB_classifier.predict(X_test.toarray())
	# print(sklearn.metrics.classification_report(y_test,pred_GNB))

	# print("Random Forest Classifier Report: \t\t")
	# pred_RF = RF_classifier.predict(X_test)
	# print(sklearn.metrics.classification_report(y_test,pred_RF))

	# print("Linear Support Vector Classifier Report: \t\t")
	# pred_LSV = LSV_classifier.predict(X_test)
	# print(sklearn.metrics.classification_report(y_test,pred_LSV))


# 	parameter_space = {
# 	'max_iter':[200,1000],    
# 	'hidden_layer_sizes': 
# 	[
# 		(500,),
# 		(750,),
# 		(1000,)
# 	#(50,50,50,50,50,50,50,50,50,50,50,50,50,50,50), 
# 	#(500,500,500),  
# 	#(10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10),
# 	#(64,64,64,64,64,64,64,64), 
# 	#(128,128,128,128,128,128,128,128),
# 	#(10,10,100,50,20,10),
# 	#(80,80,80,80,80,80,80,80),
# 	#(32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32),
# 	#(10,10,10),
# 	#(256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256),
# 	#(256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,256,256,256,256),
# 	#(256,256,256,256,256,256,256,256,256),
# 	#(256,256,256,256,256,256,256,256,256,256,256,256,128,56,10,6),
# 	#(128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128),
# 	#(500,500)
# 	#(200,10,10,10,10,10,10,10,10,200),
# 	#(10,200,200,10)                                                                                                            
# 	],
# 	#'activation':['relu'], #'tanh','identity', 'logistic'
# 	#'solver':['adam'], #'sgd','lbfgs'
# 	#'alpha':[.0001], #.0001,.0000000001,.0002,.0007,.0000000000000000000000000000000000000000000000000000000000000000000000001,.05
# 	#'learning_rate':['constant'] #'adaptive','constant','invscaling'
# }

# 	clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=2,verbose = 100)
# 	clf.fit(X_train, y_train)


# 	# Best paramete set
# 	print('Best parameters found:\n', clf.best_params_)

# 	# All results
# 	means = clf.cv_results_['mean_test_score']
# 	stds = clf.cv_results_['std_test_score']
# 	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
# 		print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))



	# STOPWORDS.add('go')
	# STOPWORDS.add('get')
	# #print(len(vectorizor.get_feature_names()))
	# mask2 = np.array(Image.open("McD_logo.png").convert("RGB"))
	# print(mask2.shape)
	# mask2[mask2 == 0] = 255
	# #mask3 = np.where(mask2 ==0,255,mask2)
	# image_colors = ImageColorGenerator(mask2)
	# tframe = pd.DataFrame(v.toarray(),columns = vectorizor.get_feature_names())
	# #print(tframe)
	# tframe.drop(labels = ['go','get'], axis = 1, inplace =True)
	# tframe = tframe.T.mean(axis=1)
	# #print(tframe)
	# # tframe = tframe != 'go'
	# # tframe = tframe != 'get'
	# # df = df[df['lema_text'] != '']
	# Cloud = WordCloud(background_color= (218,41,28), max_words=200,stopwords=STOPWORDS,mask = mask2,color_func=lambda *args, **kwargs: (255,199,44),collocations=False,height = 1600,width =1000).generate_from_frequencies(tframe) #height=800,width=1600 mode='RGBA' colormap='autumn'
	# plt.figure(figsize = (20, 10)) #facecolor = 'k'
	# plt.imshow(Cloud,interpolation="bilinear") #.recolor(color_func=image_colors)
	# plt.axis("off")
	# #plt.tight_layout(pad = 0)

	# plt.show()





##### MLP OPTIMIZED CHARTS

pred = mlp2.predict(X_test)

CM = confusion_matrix(y_test,pred)
cm_display = ConfusionMatrixDisplay(CM)
cm_display.plot()
plt.savefig('Confusion_Matrix.png')


y_score = mlp2.predict_proba(X_test)


prec, recall, _ = precision_recall_curve(y_test, y_score[:,1], pos_label=4)
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)
pr_display.plot()
plt.savefig('precison_recall.png')


fpr, tpr, _ = roc_curve(y_test, y_score[:,1], pos_label=4)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
roc_display.plot()
plt.savefig('Roc_Curve.png')


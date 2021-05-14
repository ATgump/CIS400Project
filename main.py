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

if __name__ == "__main__":
	pd.set_option('display.max_rows', 500)
	n = random.randint(0, 2**32 - 20)

	client = pymongo.MongoClient('mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true')
	db = client['tweet_DB']


	df = pd.DataFrame(list(db['processed_Training_Data_Three'].find({},{'_id':0,'lema_text':1,'label':1})))

## CREATE AND SAVE TFIDF VECTORIZOR ### 
	vectorizor = sk.text.TfidfVectorizer(min_df=.000027) #stop_words=sk._stop_words.ENGLISH_STOP_WORDS,min_df=.00003,max_df=.94
	v = vectorizor.fit_transform(df['lema_text'].to_numpy())
	TF_Vec = 'TFIDF_Vectorizer.sav'
	joblib.dump(vectorizor, TF_Vec)

	print("Random state = " + str(n))
	X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(v,df['label'],test_size=.2,random_state=2372017502) #tframe.drop('CLASS_ATR',axis = 1),tframe['CLASS_ATR'] 1016102622 2372017502
	
	### TRAIN AND SAVE MODELS/Reports ### 

	# MNB_classifier=sklearn.naive_bayes.MultinomialNB()
	# MNB_classifier.fit(X_train,y_train)
	# MNB = 'Multinomial_Naive_Bayes_Trained_Model.sav'
	# joblib.dump(MNB_classifier, MNB)
	
	# print("Multinomial Naive Bayes Classifier Report: \t\t")
	# pred_MNB = MNB_classifier.predict(X_test)
	# report = sklearn.metrics.classification_report(y_test,pred_MNB)
	# print(report)
	# with open('MNB_Table.txt','w') as file:
	# 	file.write(report)




	# mlp = MLPClassifier()
	# mlp.fit(X_train,y_train)
	# DMLPC = 'Default_Multi_Layer_Perceptron_Trained_Model.sav'
	# joblib.dump(mlp, DMLPC)

	# print("Multi-Layer Perceptron Classifier Report Default: \t\t")
	# pred_MLP = mlp.predict(X_test)
	# report = sklearn.metrics.classification_report(y_test,pred_MLP)
	# print(report)
	# with open('Default_MLP_Table.txt','w') as file:
	# 	file.write(report)




	# mlp2 = MLPClassifier(hidden_layer_sizes = (1000,)) 
	# mlp2.fit(X_train,y_train)
	# MLPC = 'Optimized_Multi_Layer_Perceptron_Trained_Model.sav'
	# joblib.dump(mlp2, MLPC)

	# print("Multi-Layer Perceptron Classifier Report Optimized: \t\t")
	# pred_MLP2 = mlp2.predict(X_test)
	# report = sklearn.metrics.classification_report(y_test,pred_MLP2)
	# print(report)
	# with open('Optimized_MLP_Table.txt','w') as file:
	# 	file.write(report)



	# SV_classifier = svm.SVC()
	# SV_classifier.fit(X_train,y_train)
	# SV = 'Support_Vector_Trained_Model.sav'
	# joblib.dump(SV_classifier, SV)

	# print("Support Vector Classifier Report: \t\t")
	# pred_SV = SV_classifier.predict(X_test)
	# report = sklearn.metrics.classification_report(y_test,pred_SV)
	# print(report)
	# with open('SV_Table.txt','w') as file:
	# 	file.write(report)




	# DT_classifier = DecisionTreeClassifier()
	# DT_classifier.fit(X_train,y_train)
	# DT = 'Decision_Tree_Trained_Model.sav'
	# joblib.dump(DT_classifier, DT)

	# print("Decision Tree Classifier Report: \t\t")
	# pred_DT = DT_classifier.predict(X_test)
	# report = sklearn.metrics.classification_report(y_test,pred_DT)
	# print(report)
	# with open('DT_Table.txt','w') as file:
	# 	file.write(report)




	# GNB_classifier = GaussianNB()
	# GNB_classifier.fit(X_train.toarray(),y_train)
	# GNB = 'Gaussian_Naive_Bayes_Trained_Model.sav'
	# joblib.dump(GNB_classifier, GNB)

	# print("Gausian Naive Bayes Classifier Report: \t\t")
	# pred_GNB = GNB_classifier.predict(X_test.toarray())
	# report = sklearn.metrics.classification_report(y_test,pred_GNB)
	# print(report)
	# with open('GNB_Table.txt','w') as file:
	# 	file.write(report)



	# RF_classifier = RandomForestClassifier()
	# RF_classifier.fit(X_train,y_train)
	# RFC = 'Random_Forest_Trained_Model.sav'
	# joblib.dump(RF_classifier, RFC)

	# print("Random Forest Classifier Report: \t\t")
	# pred_RF = RF_classifier.predict(X_test)
	# report = sklearn.metrics.classification_report(y_test,pred_RF)
	# print(report)
	# with open('RF_Table.txt','w') as file:
	# 	file.write(report)




	# LSV_classifier = LinearSVC()
	# LSV_classifier.fit(X_train,y_train)
	# LSVC = 'Linear_Support_Vector_Trained_Model.sav'
	# joblib.dump(LSV_classifier, LSVC)
	
	# print("Linear Support Vector Classifier Report: \t\t")
	# pred_LSV = LSV_classifier.predict(X_test)
	# report = sklearn.metrics.classification_report(y_test,pred_LSV)
	# print(report)
	# with open('LSV_Table.txt','w') as file:
	# 	file.write(report)

### OPTIMIZE GRID SEARCH ### 

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



### CREATE WORD CLOUD ###

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

# pred = mlp2.predict(X_test)

# CM = confusion_matrix(y_test,pred)
# cm_display = ConfusionMatrixDisplay(CM)
# cm_display.plot()
# plt.savefig('Confusion_Matrix.png')


# y_score = mlp2.predict_proba(X_test)


# prec, recall, _ = precision_recall_curve(y_test, y_score[:,1], pos_label=4)
# pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)
# pr_display.plot()
# plt.savefig('precison_recall.png')


# fpr, tpr, _ = roc_curve(y_test, y_score[:,1], pos_label=4)
# roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
# roc_display.plot()
# plt.savefig('Roc_Curve.png')


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
#mongo connection
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


    #print("Random state = " + str(n))
    X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(v,df['label'],test_size=.2,random_state=n) #tframe.drop('CLASS_ATR',axis = 1),tframe['CLASS_ATR']
    
    
    # MNB_classifier=sklearn.naive_bayes.MultinomialNB()
    # MNB_classifier.fit(X_train,y_train)

    mlp = MLPClassifier(max_iter=500) #max_iter=250
    # mlp.fit(X_train,y_train)

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

#     print("Multinomial Naive Bayes Classifier Report: \t\t")
#     pred_MNB = MNB_classifier.predict(X_test)
#     print(sklearn.metrics.classification_report(y_test,pred_MNB))

    # print("Multi-Layer Perceptron Classifier Report: \t\t")
    # pred_MLP = mlp.predict(X_test)
    # print(sklearn.metrics.classification_report(y_test,pred_MLP))

    # print("Support Vector Classifier Report: \t\t")
    # pred_SV = SV_classifier.predict(X_test)
    # print(sklearn.metrics.classification_report(y_test,pred_SV))
# ###
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


    parameter_space = {
    'hidden_layer_sizes': 
    [
    #(50,50,50,50,50,50,50,50,50,50,50,50,50,50,50), 
    #(500,500,500),  
    #(10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10),
    #(64,64,64,64,64,64,64,64), 
    #(128,128,128,128,128,128,128,128),
    #(10,10,100,50,20,10),
    #(80,80,80,80,80,80,80,80),
    #(32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32),
    #(10,10,10),
    #(256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256),
    #(256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,256,256,256,256),
    (256,256,256,256,256,256,256,256,256),
    #(256,256,256,256,256,256,256,256,256,256,256,256,128,56,10,6),
    #(128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128),
    #(500,500)
    #(200,10,10,10,10,10,10,10,10,200),
    #(10,200,200,10)                                                                                                            
    ],
    'activation':['relu'], #'tanh','identity', 'logistic'
    'solver':['adam'], #'sgd','lbfgs'
    'alpha':[.0001], #.0001,.0000000001,.0002,.0007,.0000000000000000000000000000000000000000000000000000000000000000000000001,.05
    'learning_rate':['adaptive','constant','invscaling'] #'constant','invscaling'
}

    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3,verbose = 100)
    clf.fit(X_train, y_train)


    # Best paramete set
    print('Best parameters found:\n', clf.best_params_)

    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))



    
    #print(len(vectorizor.get_feature_names()))

    # tframe = pd.DataFrame(v.toarray(),columns = vectorizor.get_feature_names())
    # tframe = tframe.T.sum(axis=1)
    # Cloud = WordCloud(background_color="white", max_words=100,stopwords=STOPWORDS).generate_from_frequencies(tframe)
    # plt.figure(figsize = (8, 8), facecolor = None)
    # plt.imshow(Cloud)
    # plt.axis("off")
    # plt.tight_layout(pad = 0)

    # plt.show()


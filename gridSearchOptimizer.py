import joblib
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd
import pymongo

if __name__ == "__main__":
	
	## Load data and vectorizor, construct model, fit vectorizor
	client = pymongo.MongoClient('mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true')
	db = client['tweet_DB']
	df = pd.DataFrame(list(db['processed_Training_Data'].find({},{'_id':0,'lema_text':1,'label':1})))
	vec = joblib.load('.\\trained_Models\\TFIDF_Vectorizer.sav')
	v = vec.transform(df['lema_text'].to_numpy())
	X_train,X_test,y_train,y_test = train_test_split(v,df['label'],test_size=.2,random_state=2372017502)
	mlp = MLPClassifier()
	
	## Parameters to search this example may
	parameter_space = {
	#'max_iter':[200,1000],    
	'hidden_layer_sizes': 
	[
		(500,),
		#(750,),
		(1000,)
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
	#(256,256,256,256,256,256,256,256,256),
	#(256,256,256,256,256,256,256,256,256,256,256,256,128,56,10,6),
	#(128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128),
	#(500,500)
	#(200,10,10,10,10,10,10,10,10,200),
	#(10,200,200,10)                                                                                                            
	],
	#'activation':['relu'], #'tanh','identity', 'logistic'
	#'solver':['adam'], #'sgd','lbfgs'
	#'alpha':[.0001], #.0001,.0000000001,.0002,.0007,.0000000000000000000000000000000000000000000000000000000000000000000000001,.05
	#'learning_rate':['constant'] #'adaptive','constant','invscaling'
}

	## Begin validation
	clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3,verbose = 100)
	clf.fit(X_train, y_train)


	# Best parameters set
	print('Best parameters found:\n', clf.best_params_)

	# Show All results
	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

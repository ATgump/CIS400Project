from joblib import Parallel, delayed
import joblib
from tweetPreprocessor import batch_lemmatizer
import pymongo
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

# return a generator for the chunks
def chunker(tweet_list,length,chunksize):
	return(tweet_list[pos:pos + chunksize] for pos in range(0,length,chunksize)) 

# flatten list of lists into list of labeled tweets
def flatten(L):
	return [tweet for batch in L for tweet in batch]

## Run experiment (Comment out the different tests to generate labels using different models)
def chunk_processor(texts,method):
	preproc_pipe=[]

	## Load model(s)
	Vectorizor = joblib.load('.\\trained_Models\\TFIDF_Vectorizer.sav')
	MLP_model = joblib.load('.\\trained_Models\\Optimized_Multi_Layer_Perceptron_Trained_Model.sav')
	RF_model = joblib.load('.\\trained_Models\\Random_Forest_Trained_Model.sav')
	LSV_model = joblib.load('.\\trained_Models\\Linear_Support_Vector_Trained_Model.sav')

# Test1: Multi-Layer Perceptron Optimized
	if method == 'MLP':
		for twt in texts:
			test = Vectorizor.transform([twt])
			if MLP_model.predict(test) == 0:
				preproc_pipe.append({'text':twt,'label':0})
			elif MLP_model.predict(test) == 4: 				
				preproc_pipe.append({'text':twt,'label':4})
			elif MLP_model.predict(test) == 2: 				
				preproc_pipe.append({'text':twt,'label':2})

	
# Test2: Random Forest
	elif method == 'RF':	
		for twt in texts:
			test = Vectorizor.transform([twt])
			if RF_model.predict(test) == 0:
				preproc_pipe.append({'text':twt,'label':0})
			elif RF_model.predict(test) == 4: 				
				preproc_pipe.append({'text':twt,'label':4})
			elif RF_model.predict(test) == 2: 				
				preproc_pipe.append({'text':twt,'label':2})

#Test3: Linear Support Vector
	elif method == 'LSV':	
		for twt in texts:
			test = Vectorizor.transform([twt])
			if LSV_model.predict(test) == 0:
				preproc_pipe.append({'text':twt,'label':0})
			elif LSV_model.predict(test) == 4: 				
				preproc_pipe.append({'text':twt,'label':4})
			elif LSV_model.predict(test) == 2: 				
				preproc_pipe.append({'text':twt,'label':2})


##Test4: Combined	
	elif method == 'Combined':
		for twt in texts:
			test = Vectorizor.transform([twt])
			if LSV_model.predict(test) == 0 and RF_model.predict(test) == 0 and MLP_model.predict(test) == 0:
				preproc_pipe.append({'text':twt,'label':0})
			elif LSV_model.predict(test) == 4 and RF_model.predict(test) == 4 and MLP_model.predict(test) == 4: 				
				preproc_pipe.append({'text':twt,'label':4})
			elif LSV_model.predict(test) == 2 and RF_model.predict(test) == 2 and MLP_model.predict(test) == 2: 				
				preproc_pipe.append({'text':twt,'label':2})

	return preproc_pipe

## Label tweets in batches
def batch_labeler(texts,chunksize=100, method = None):
	executor = Parallel(n_jobs=-1, backend='multiprocessing', prefer="processes")
	do = delayed(chunk_processor)
	tasks = (do(chunk,method) for chunk in chunker(texts, len(texts), chunksize=chunksize))
	result = executor(tasks)
	return flatten(result)

if __name__ == "__main__":

	## Get Experiment data to label
	mConnection = 'mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true'
	client = pymongo.MongoClient(mConnection)
	db = client['tweet_DB']
	restaurants = ['Wendys', 'BK', 'McD']
	methods = ['Combined', 'MLP', 'RF', 'LSV']
	
	for restaurant in restaurants:
		if restaurant == 'McD':
			col = 'experimental_Data_McDonalds'
		else:
			col =  'experimental_Data_' + restaurant
		cursor = db[col].find({},{'text':1,'_id':0})
		df = pd.DataFrame(list(cursor))
		print(df)
		df['lema_text'] = (pd.Series(batch_lemmatizer(df['text'],50)))
		df = df[df['lema_text'] != '']
		print(df)
		for method in methods:
			ins = 'exp_Data_' + restaurant + '_' + method
			labeled = batch_labeler(df['lema_text'],chunksize=128,method = method)
			print(pd.DataFrame(labeled))
			db[ins].insert_many(labeled)

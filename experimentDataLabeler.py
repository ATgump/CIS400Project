from nltk.sentiment.vader import SentimentIntensityAnalyzer
from joblib import Parallel, delayed
from textblob import TextBlob
import joblib
from ourCorpus import batch_lemmatizer
import pymongo
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn.neural_network import MLPClassifier

def chunker(tweet_list,length,chunksize):
	return(tweet_list[pos:pos + chunksize] for pos in range(0,length,chunksize)) # return a generator for the chunks

def flatten(L):
	return [tweet for batch in L for tweet in batch]

def chunk_processor(texts):
	preproc_pipe=[]
	Vectorizor = joblib.load('TFIDF_Vectorizer.sav')
	#print(type(Vectorizor))
	MLP_model = joblib.load('Multi_Layer_Perceptron_Trained_Model.sav')
	RF_model = joblib.load('Random_Forest_Trained_Model.sav')
	LSV_model = joblib.load('Linear_Support_Vector_Trained_Model.sav')


	# for twt in texts:
	# 	test = Vectorizor.transform([twt])
	# 	if MLP_model.predict(test) == 0:
	# 		preproc_pipe.append({'text':twt,'label':0})
	# 	elif MLP_model.predict(test) == 4: 				
	# 		preproc_pipe.append({'text':twt,'label':4})
	# 	elif MLP_model.predict(test) == 2: 				
	# 		preproc_pipe.append({'text':twt,'label':2})
	# 	else:
	# 		print("This is a bug")
	
	
	# for twt in texts:
	# 	test = Vectorizor.transform([twt])
	# 	if RF_model.predict(test) == 0:
	# 		preproc_pipe.append({'text':twt,'label':0})
	# 	elif RF_model.predict(test) == 4: 				
	# 		preproc_pipe.append({'text':twt,'label':4})
	# 	elif RF_model.predict(test) == 2: 				
	# 		preproc_pipe.append({'text':twt,'label':2})
	# 	else:
	# 		print("This is a bug")
	
	
	# for twt in texts:
	# 	test = Vectorizor.transform([twt])
	# 	if LSV_model.predict(test) == 0:
	# 		preproc_pipe.append({'text':twt,'label':0})
	# 	elif LSV_model.predict(test) == 4: 				
	# 		preproc_pipe.append({'text':twt,'label':4})
	# 	elif LSV_model.predict(test) == 2: 				
	# 		preproc_pipe.append({'text':twt,'label':2})
	# 	else:
	# 		print("This is a bug")

	
	for twt in texts:
		test = Vectorizor.transform([twt])
		if LSV_model.predict(test) == 0 and RF_model.predict(test) == 0 and MLP_model.predict(test) == 0:
			preproc_pipe.append({'text':twt,'label':0})
		elif LSV_model.predict(test) == 4 and RF_model.predict(test) == 4 and MLP_model.predict(test) == 4: 				
			preproc_pipe.append({'text':twt,'label':4})
		elif LSV_model.predict(test) == 2 and RF_model.predict(test) == 2 and MLP_model.predict(test) == 2: 				
			preproc_pipe.append({'text':twt,'label':2})
		# else:
		# 	print("This is a bug")


	return preproc_pipe

def batch_labeler(texts,chunksize=100):
	executor = Parallel(n_jobs=4, backend='multiprocessing', prefer="processes")
	do = delayed(chunk_processor)
	tasks = (do(chunk) for chunk in chunker(texts, len(texts), chunksize=chunksize))
	result = executor(tasks)
	return flatten(result)

if __name__ == "__main__":
	mConnection = 'mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true'
	client = pymongo.MongoClient(mConnection)
	db = client['tweet_DB']
	cursor = db['experimental_Data_McDonalds'].find({},{'text':1,'_id':0})
	df = pd.DataFrame(list(cursor))
	print(df)
	df['lema_text'] = (pd.Series(batch_lemmatizer(df['text'],50)))
	df = df[df['lema_text'] != '']
	print(df)
	labeled = batch_labeler(df['lema_text'],chunksize=128 )
	print(pd.DataFrame(labeled))
	
	### JUST FOR INSERTING (leave commented) ##
	db['exp_Data_McD_Combined'].insert_many(labeled)


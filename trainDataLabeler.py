from nltk.sentiment.vader import SentimentIntensityAnalyzer
from joblib import Parallel, delayed
from textblob import TextBlob

def chunker(tweet_list,length,chunksize):
	return(tweet_list[pos:pos + chunksize] for pos in range(0,length,chunksize)) # return a generator for the chunks

def flatten(L):
	return [tweet for batch in L for tweet in batch]

def chunk_processor(texts):
	preproc_pipe=[]
	se = SentimentIntensityAnalyzer()
	for twt in texts:
		tb = TextBlob(twt)
		if se.polarity_scores(twt)['compound'] < 0 and tb.sentiment.polarity < 0:
			preproc_pipe.append({'text':twt,'label':'neg'})
		if se.polarity_scores(twt)['compound'] > 0 and tb.sentiment.polarity > 0: 				
			preproc_pipe.append({'text':twt,'label':'pos'})
		if se.polarity_scores(twt)['compound'] == 0 and tb.sentiment.polarity == 0: 				
			preproc_pipe.append({'text':twt,'label':'neut'})
	return preproc_pipe

def batch_labeler(texts,chunksize=100):
	executor = Parallel(n_jobs=6, backend='multiprocessing', prefer="processes")
	do = delayed(chunk_processor)
	tasks = (do(chunk) for chunk in chunker(texts, len(texts), chunksize=chunksize))
	result = executor(tasks)
	return flatten(result)

if __name__ == "__main__":
	mConnection = 'mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true'
	client = pymongo.MongoClient(mConnection)
	db = client['tweet_DB']
	cursor = db['unprocessed_Tweets'].find({},{'text':1,'_id':0})
	df = pd.DataFrame(list(cursor))
	print(df)
	labeled = batch_labeler(df['text'],chunksize=256 )
	print(pd.DataFrame(labeled))
	
	### JUST FOR INSERTING (leave commented) ##
	#db['labeled_Training_Data'].insert_many(labeled)
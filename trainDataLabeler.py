


from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from joblib import Parallel, delayed




def chunker(tweet_list,length,chunksize):
	return(tweet_list[pos:pos + chunksize] for pos in range(0,length,chunksize)) #return a generator for the chunks

def flatten(L):
	return [tweet for batch in L for tweet in batch]

def chunk_processor(texts):
	preproc_pipe=[]
	sentiment_Analyser = pipeline("sentiment-analysis")
	se = SentimentIntensityAnalyzer()
	for twt in texts:
		if se.polarity_scores(twt)['compound'] < 0 and sentiment_Analyser(twt)[0]['label'] == 'NEGATIVE':
			preproc_pipe.append({'text':twt,'label':'neg'})
		if se.polarity_scores(twt)['compound'] > 0 and sentiment_Analyser(twt)[0]['label'] == 'POSITIVE':
			preproc_pipe.append({'text':twt,'label':'pos'})
	return preproc_pipe

def batch_labeler(texts,chunksize=100):
	executor = Parallel(n_jobs=4, backend='multiprocessing', prefer="processes")
	do = delayed(chunk_processor)
	tasks = (do(chunk) for chunk in chunker(texts, len(texts), chunksize=chunksize))
	result = executor(tasks)
	return flatten(result)

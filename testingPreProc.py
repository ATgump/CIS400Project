import ourCorpus
import numpy as np
import pandas as pd
import trainDataLabeler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import pymongo
import myTwitterCookbook
#textblob/VADER do nuetral wheras transformers do not do we want transformers or just maybe use transformers to test against as
#a benchmark
if __name__ == "__main__":
	mConnection = 'mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true'
	test_set = ["I hate tensorflow", "I hate burgers","I hate frys","I think you're stupid",\
		"I dont like game of thrones", "I love milk", "I like cows","i think this song is good",\
			"The boys is a good show","Mcdonalds is a great restaurant","test","neutral tweet"]
	client = pymongo.MongoClient('mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true')
	db = client['tweet_DB']
	cursor = db['unprocessed_Tweets'].find({},{'text':1,'_id':0})
	df = pd.DataFrame(list(cursor))
	print(df)
	labeled = trainDataLabeler.batch_labeler(df['text'])
	print(pd.DataFrame(labeled))
	#ids = myTwitterCookbook.save_to_mongo(labeled, 'tweet_DB', 'testing_Tweets_Labeled', host=mConnection)
	# test = trainDataLabeler.batch_labeler(test_set)
	# df = pd.DataFrame(test)
	# print(df)








	# labeled_data = []
	# sentiment_Analyser = pipeline("sentiment-analysis")
	# print(sentiment_Analyser("Mcdonalds is a great restaurant"))
	# se = SentimentIntensityAnalyzer()
	# print(se.polarity_scores("Mcdonalds is a great restaurant"))
	# if se.polarity_scores("I hate tensorflow")['compound'] < 0 and sentiment_Analyser("I hate tensorflow")[0]['label'] == 'NEGATIVE':
	# 	labeled_data.append({'I hate tensorflow':0})
	# 	print("appended")
	
	# print(labeled_data)
	
	
	
	
	
	
	
	
	
	
	# test = "This is :( my test :D string"
	# test2 = "This is another test"

	# print((pd.Series(ourCorpus.batch_lemmatizer([test,test2],100))))

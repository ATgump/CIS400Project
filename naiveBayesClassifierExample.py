#import sys
#sys.path.insert(0,r'C:\Users\avery\Desktop\CIS 400\MyTwitterCookbook\myTwitterCookbook')  ##  testing importing from specific directories
import myTwitterCookbook
import ourCorpus
import spellchecker
from nltk.corpus import twitter_samples as tweets
#from nltk.text import TextCollection, Text
from nltk.classify.naivebayes import NaiveBayesClassifier as NBC
import json
import pymongo ## pymongo is used to interface with mongo client
## Mongo connection (my database with your user login)
client = pymongo.MongoClient('mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true') 
login = myTwitterCookbook.oauth_login()
db = client.FirstDB ## create a database obj (databases consist of tables and each table has k:v pairs)
db.firstCollection.insert_one({'y':1}) ## firstCollection is a table and the k:v  -  'y' : 1 is added to the table tables (and databases) are created when they are first added to 


# follow install instructions for twitter_samples corpus from: https://www.nltk.org/data.html
pos_tweets = tweets.strings('positive_tweets.json')
neg_tweets = tweets.strings('negative_tweets.json')

print('len of pos_tweets: ' + str(len(pos_tweets)))
# create two corpus' (from ourCorpus) one for the positive training tweets and one for the negative
corp_pos = ourCorpus.corpus(pos_tweets,len(pos_tweets)) 
corp_neg = ourCorpus.corpus(neg_tweets,len(neg_tweets))
print('len of corp_pos: ' + str(len(corp_pos.doc_list)))
# generate tf-idf vectors for the data (uses my implementaion we should switch to scikit because it also has multinomial NBC which is better than this nltk one for discrete feature values (tfidf))
pos_train_data = corp_pos.TF_IDF_feature_vector()
neg_train_data = corp_neg.TF_IDF_feature_vector()
print('len of pos_train: ' + str(len(pos_train_data)))

print(json.dumps(pos_train_data))


#create training data then train the NBC classifier
train_data = [(v,1) for v in pos_train_data] + [(v,0) for v in neg_train_data]
print(json.dumps(train_data,indent = 2))
classifier = NBC.train(train_data) 

tweet = "i feel negative, i dont like movies."
tweet2 = "this day is nice i feel positive right now"
tweet3 = "i am happy"
tweet4 = "i am sad"
corp2 = ourCorpus.corpus([tweet,tweet2,tweet3,tweet4], 4)
features = corp2.TF_IDF_feature_vector()

for v in features:
    print(str(classifier.classify(v)))

















#corp = ourCorpus.corpus()
# 0 - doesnt take covid seriously, 1 - neutral, 2 - takes covid seriously
# sample_data = {
#     "COVID19 is fake news":0,
#     "We need COVID19 relief":2, 
#     "I like my cat":1,
#     "who cares about COVID virus":0,
#     "I like what Biden has been doing with covid":2,
#     "Go packers":1,
#     "I hate biden theres no such thing as covid":0,
# }


# feature_word_set = set()
# for k in sample_data:
#     l = k.lower().split()
#     for w in l:
#         feature_word_set.add(w)
#print(feature_word_set)

# def features(tweet):
#     words = tweet.lower().split()
#     return {'Contains(%s)'%w:(w in words) for w in feature_word_set} ## instead of feature(words) values being True/False make them TF-IDF values

## attempt to get nltk tf-idf working ##
# dic1 = dict()
# for thing in coll:
#     dic1[thing] = coll.tf_idf(thing,coll)

# dic2 = {normalize_tweet(k):dic1.get(k) for k in dic1}
# print(json.dumps(dic2, indent=1))
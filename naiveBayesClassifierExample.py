#import sys
#sys.path.insert(0,r'C:\Users\avery\Desktop\CIS 400\MyTwitterCookbook\myTwitterCookbook')  ##  testing importing from specific directories
import myTwitterCookbook
from nltk.classify.naivebayes import NaiveBayesClassifier as NBC
import json
import pymongo ## pymongo is used to interface with mongo client
## Mongo connection (my database with your user login)
client = pymongo.MongoClient('mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true') 
login = myTwitterCookbook.oauth_login()

db = client.FirstDB ## create a database obj (databases consist of tables and each table has k:v pairs)
db.firstCollection.insert_one({'y':1}) ## firstCollection is a table and the k:v  -  'y' : 1 is added to the table tables (and databases) are created when they are first added to 

# 0 - doesnt take covid seriously, 1 - neutral, 2 - takes covid seriously
sample_data = {
    "COVID19 is fake news":0,
    "We need COVID19 relief":2, 
    "I like my cat":1,
    "who cares about COVID virus":0,
    "I like what Biden has been doing with covid":2,
    "Go packers":1,
    "I hate biden theres no such thing as covid":0,
}


feature_word_set = set()
for k in sample_data:
    l = k.lower().split()
    for w in l:
        feature_word_set.add(w)
print(feature_word_set)

def features(tweet):
    words = tweet.lower().split()
    return {'Contains(%s)'%w:(w in words) for w in feature_word_set} ## instead of feature(words) values being True/False make them TF-IDF values

#create training data then train the NBC classifier
test_data = [(features(k),v) for (k,v) in sample_data.items()]
print(json.dumps(test_data, indent = 2))
classifier = NBC.train(test_data) 

tweet = "this is a test"
tweet2 = "Covid isn't real biden doesn't know what hes talking about"
tweet3 = "Lets solve COVID together masks are great"


print(str(classifier.classify(features(tweet))))
print(str(classifier.classify(features(tweet2))))
print(str(classifier.classify(features(tweet3))))

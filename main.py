import myTwitterCookbook
import pandas as pd
import numpy as np
import pymongo
import json
import sklearn
import sklearn.naive_bayes
from sklearn import feature_extraction as sk
from nltk.corpus import twitter_samples as tweets
import ourCorpus
#mongo connection
client = pymongo.MongoClient('mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true')
db = client['test_Search_Mining_DB']

#get the list[{full_text : tweet}]
cursor = db['T'].find({},{'extended_tweet':{'full_text':1},'_id':0})


neg_tweets = tweets.strings("negative_tweets.json")
pos_tweets = tweets.strings("positive_tweets.json")
neg_tweets = [ourCorpus.normalize_tweet(docs) for docs in neg_tweets]
neg_tweets = [docs for docs in neg_tweets if not docs.isspace()]
pos_tweets = [ourCorpus.normalize_tweet(docs) for docs in pos_tweets]
pos_tweets = [docs for docs in pos_tweets if not docs.isspace()]
raw_docs = neg_tweets + pos_tweets
# raw_docs = [ourCorpus.normalize_tweet(docs) for docs in raw_docs]
# raw_docs = [docs for docs in raw_docs if not docs.isspace()]

train_d = [dict({'text':neg,'class':0}) for neg in neg_tweets] + [dict({'text':pos,'class':1}) for pos in pos_tweets]
frame = pd.DataFrame(train_d)
#print(frame)

testframe = (frame.drop('text',axis =1)).to_numpy()
print(testframe)

#l = [et['extended_tweet'] for et in list(cursor)]
#l2 = [ourCorpus.normalize_tweet(et['full_text']) for et in l]

## fit to matrix
#min_df = 0.01,max_df=0.85
#min_df = 0.0002,max_df=.99
print(raw_docs)
cv = sk.text.CountVectorizer(stop_words=sk._stop_words.ENGLISH_STOP_WORDS,token_pattern=r'[^\s]+') #token patern matches emoticons now too
word_count_vector = cv.fit_transform(raw_docs)
print(cv.get_feature_names())
tfidf_transformer = sk.text.TfidfTransformer(smooth_idf = True, use_idf = True)
test = tfidf_transformer.fit_transform(word_count_vector)
tframe = pd.DataFrame(test.toarray(),columns = cv.get_feature_names())
tframe['CLASS_ATR'] = testframe
print(tframe)
X = tframe.drop('CLASS_ATR',axis = 1)
y = tframe['CLASS_ATR']


X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y,test_size=.2,random_state=42)
NB_classifier=sklearn.naive_bayes.MultinomialNB()
NB_classifier.fit(X_train,y_train)
pred_NB = NB_classifier.predict(X_test)
print(sklearn.metrics.classification_report(y_test,pred_NB))


#print("THIS IS L2\n\n" + str(l2) + "\n\n")

# print(pd.DataFrame(mat))
# print(mat[0][0].toarray())
#col = myTwitterCookbook.load_from_mongo('test_Search_Mining_DB', 'T', q, host=client)
# for row in mat:
#     for entry in row:
#         print(entry[1])






#vectorizor = sk.text.TfidfVectorizer()
#vectorizor = sk.text.TfidfVectorizer(max_features=100, stop_words='english')
#mat2 = vectorizor.fit_transform(raw_documents = raw_docs)
#print(vectorizor.get_feature_names())
#neg_mat = vectorizor.transform(raw_documents = neg_tweets)
#pos_mat = vectorizor.transform(raw_documents = pos_tweets)
#tokenizer = vectorizor.build_tokenizer()
#train_d = mat2
#train_d = [dict({'text':neg,'class':0}) for neg in neg_mat] + [dict({'text':pos,'class':1}) for pos in pos_mat]
#frame = pd.DataFrame(columns = vectorizor.get_feature_names())
#frame2 = pd.DataFrame(train_d)
#mat = vectorizor.fit_transform(raw_documents = l2)




#print(json.dumps(raw_docs, indent = 1)+"\n\n\n")
# raw_docs = [ourCorpus.normalize_tweet(docs) for docs in raw_docs]
# raw_docs = [docs for docs in raw_docs if not docs.isspace()]
#print(json.dumps(raw_docs, indent = 1)+"\n\n\n")
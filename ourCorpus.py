import math
import symspellpy
import pkg_resources
from symspellpy import SymSpell
from nltk.corpus import words,stopwords
import textblob
import re
import datetime
from sklearn import feature_extraction as sk
from bs4 import BeautifulSoup
import pymongo
import pandas as pd
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

from joblib import Parallel, delayed
import spacy
from emoji import demojize
def load_dict_smileys():
	
	return {
		":‑)":"smiley",
		":-]":"smiley",
		":-3":"smiley",
		":->":"smiley",
		"8-)":"smiley",
		":-}":"smiley",
		":)":"smiley",
		":]":"smiley",
		":3":"smiley",
		":>":"smiley",
		"8)":"smiley",
		":}":"smiley",
		":o)":"smiley",
		":c)":"smiley",
		":^)":"smiley",
		"=]":"smiley",
		"=)":"smiley",
		":-))":"smiley",
		":‑D":"smiley",
		"8‑D":"smiley",
		"x‑D":"smiley",
		"X‑D":"smiley",
		":D":"smiley",
		"8D":"smiley",
		"xD":"smiley",
		"XD":"smiley",
		":‑(":"sad",
		":‑c":"sad",
		":‑<":"sad",
		":‑[":"sad",
		":(":"sad",
		":c":"sad",
		":<":"sad",
		":[":"sad",
		":-||":"sad",
		">:[":"sad",
		":{":"sad",
		":@":"sad",
		">:(":"sad",
		":'‑(":"sad",
		":'(":"sad",
		":‑P":"playful",
		"X‑P":"playful",
		"x‑p":"playful",
		":‑p":"playful",
		":‑Þ":"playful",
		":‑þ":"playful",
		":‑b":"playful",
		":P":"playful",
		"XP":"playful",
		"xp":"playful",
		":p":"playful",
		":Þ":"playful",
		":þ":"playful",
		":b":"playful",
		"<3":"love"
		}


def emoji_Replacer(twt):
	tweet_list = twt.split()
	emoticons = load_dict_smileys()
	edited = [emoticons[word] if word in emoticons else word for word in tweet_list]
	tweet = ' '.join(edited)
	tweet = demojize(tweet)
	tweet = tweet.replace(":"," ")
	#print(tweet)
	return(tweet)

test_sentence = 'How am   wendy burger king I just finkding thi\'s awegsome templatnbe by @54Mr_Meyer?! Snfag your own copy: https://t.co/DZOQovcLFl @SlidesManiaSM #Free resources for #GoogleSlides &amp; #PowerPoint @test'

def chunker(tweet_list,length,chunksize):
	return(tweet_list[pos:pos + chunksize] for pos in range(0,length,chunksize)) #return a generator for the chunks
q4 =['mcdonald', 'wendy', 'burger', 'starbuck', 'king', 'pizza', 'hut', 'inonu', 'white', 'castle', 'auntie', 'news', 'popeye', 'chick', 'fila', 'taco', 'bell', 'abyss', 'dairy' ,'queen']
words = set(q4) | set(stopwords.words('english')) | set(['face'])
def lemmatize_pipe(doc):
	lemma_list = [str(tok.lemma_).lower() for tok in doc if tok.is_alpha and str(tok.lemma_).lower() not in words] #and tok.text.lower() not in stopwords tok.text.lower()
	s = ' '.join(lemma_list)
	return s	


def flatten(L):
	#print(L)
	return [tweet for batch in L for tweet in batch]

def chunk_processor(texts):
	preproc_pipe = []
	process = []
	for twt in texts:
		#print(twt)
		twt = re.sub(r"http\S+", "", twt)
		twt = re.sub(r'https?:\/\/[^\s]+(\s|$)','',twt)
		twt = re.sub(r'^https?:\/\/.*[\r\n]*', '', twt, flags=re.MULTILINE)
		twt = BeautifulSoup(twt,features="html.parser").get_text()
		twt = emoji_Replacer(twt)
		#print(twt)
		twt = re.sub(r'(@[^\s]+(\s|$))|(#[^\s]+(\s|$))|(&[^\s]+(\s|$))|[^\w\s\']|\d','',twt)# |(#[^\s]+(\s|$)) remove @ mentions, #, urls, special characters besides ' , and numbers |(https?:\/\/[^\s]+(\s|$))
		twt = re.sub(r'\s+',' ',twt)
		twt = sym_spell.lookup_compound(twt,max_edit_distance=2)[0]._term
		twt = twt.lower()
		process.append(twt)
		#print(twt)
	nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
	for doc in nlp.pipe(process, batch_size=20):
		preproc_pipe.append(lemmatize_pipe(doc))
		# processed_doc = lemmatize_pipe(doc)
		# if processed_doc:
		# 	preproc_pipe.append(processed_doc)
	return preproc_pipe

def batch_lemmatizer(texts,chunksize=100):
	executor = Parallel(n_jobs=6, backend='multiprocessing', prefer="processes")
	do = delayed(chunk_processor)
	tasks = (do(chunk) for chunk in chunker(texts, len(texts), chunksize=chunksize))
	result = executor(tasks)
	return flatten(result)


if __name__ == "__main__":
	client = pymongo.MongoClient('mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true')
	db = client['tweet_DB']
	df = pd.DataFrame(list(db['labeled_Training_Data'].find({},{'_id':0,'text':1,'label':1})))
	df['label'].replace(to_replace = 'pos',value = 4,inplace = True)
	df['label'].replace(to_replace = 'neut',value = 2,inplace = True)
	df['label'].replace(to_replace = 'neg',value = 0,inplace = True)
	df['lema_text'] = (pd.Series(batch_lemmatizer(df['text'],50)))
	df = df[df['lema_text'] != '']

	db.processed_Training_Data_Three.insert_many(df.to_dict('records'))
	# test2 = '@hiyorihere Chick-fil-A https://t.co/HBlpCzX4sq'
	# print(str(batch_lemmatizer([test2])))















#     print(batch_lemmatizer([test_sentence]))
#print(normalize_tweet(test_sentence))
# nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
# doc = nlp.pipe([test_sentence],batch_size=1)
# for docs in doc:
# 	print(docs)
# 	lemma_list = [str(tok.lemma_).lower() for tok in docs if tok.is_alpha]
# print(lemma_list)
# s = ' '.join(lemma_list)
# print(s)
# print(sym_spell.lookup_compound(s,max_edit_distance=2)[0]._term.split())
## split regexs

	# twt = re.sub(r'\s+',' ',twt) #remove extra white space
	# #print('removed white space: '+twt)
	# twt = re.sub(r'@[^\s]+(\s|$)','',twt)# remove @ mentions
	# #print('removed @ mention: '+twt)
	# twt = re.sub(r'#[^\s]+(\s|$)','',twt) #remove #tags
	# #print('removed tags: '+twt)
	# twt = re.sub(r'https?://[^\s]+(\s|$)','',twt) # remove links
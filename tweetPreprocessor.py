import pkg_resources
from symspellpy import SymSpell
from nltk.corpus import words,stopwords
import re
from bs4 import BeautifulSoup
import pymongo
import pandas as pd
from joblib import Parallel, delayed
import spacy
from emoji import demojize

## Setup Symspell ##
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

## Dictionary for emoticon conversion ## 
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

## Replace emojis and emoticons
def emoji_Replacer(twt):
	tweet_list = twt.split()
	emoticons = load_dict_smileys()
	edited = [emoticons[word] if word in emoticons else word for word in tweet_list]
	tweet = ' '.join(edited)
	tweet = demojize(tweet)
	tweet = tweet.replace(":"," ")
	return(tweet)

# Return a generator for the chunks
def chunker(tweet_list,length,chunksize):
	return(tweet_list[pos:pos + chunksize] for pos in range(0,length,chunksize)) 

# Create set of stopwords to remove 
q4 =['mcdonald', 'wendy', 'burger', 'starbuck', 'king', 'pizza', 'hut', 'inonu', 'white', 'castle', 'auntie', 'news', 'popeye', 'chick', 'fila', 'taco', 'bell', 'abyss', 'dairy' ,'queen']
words = set(q4) | set(stopwords.words('english')) | set(['face'])

#Perform Lemmatization and stop word removal
def lemmatize_pipe(doc):
	lemma_list = [str(tok.lemma_).lower() for tok in doc if tok.is_alpha and str(tok.lemma_).lower() not in words] ## remove stop words and lemmatize
	s = ' '.join(lemma_list)
	return s	

## Flatten list of lists to list of tweets
def flatten(L):
	return [tweet for batch in L for tweet in batch]

## Process Tweets in batches
def chunk_processor(texts):
	preproc_pipe = []
	process = []
	for twt in texts:
		#print(twt)
		twt = re.sub(r'https?:\/\/[^\s]+(\s|$)','',twt) ## remove URLs
		#print(twt)
		twt = BeautifulSoup(twt,features="html.parser").get_text() ## convert html tags
		#print(twt)
		twt = emoji_Replacer(twt) ## remove emojis/emoticons
		#print(twt)
		twt = re.sub(r'(@[^\s]+(\s|$))','',twt) ## remove @mentions
		#print(twt)
		twt = re.sub(r'(#[^\s]+(\s|$))','',twt) ## remove #hashtags
		#print(twt)
		twt = re.sub(r'[^\w\s\']','',twt) ## remove special characters exclude ' 
		#print(twt)
		twt = re.sub(r'\d','',twt) ## remove digits
		#print(twt)
		twt = re.sub(r'\s+',' ',twt) ## remove extra whitespace
		#print(twt)
		twt = sym_spell.lookup_compound(twt,max_edit_distance=2)[0]._term ## correct spelling errors
		#print(twt)
		twt = twt.lower() ## make lower case 
		#print(twt)
		process.append(twt)
		#print(twt)
	
	## Use spaCy pipeline for optimized lemmatizing 
	nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
	for doc in nlp.pipe(process, batch_size=20):
		preproc_pipe.append(lemmatize_pipe(doc))
	return preproc_pipe

## Batching for parallel computing
def batch_lemmatizer(texts,chunksize=100):
	executor = Parallel(n_jobs=1, backend='multiprocessing', prefer="processes")
	do = delayed(chunk_processor)
	tasks = (do(chunk) for chunk in chunker(texts, len(texts), chunksize=chunksize))
	result = executor(tasks)
	return flatten(result)

#Do preprocessing
if __name__ == "__main__":
	client = pymongo.MongoClient('mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true')
	db = client['tweet_DB']

	df = pd.DataFrame(list(db['labeled_Training_Data'].find({},{'_id':0,'text':1,'label':1})))
	print(df)
	## replace string labels with integer values 
	## (was a mistake to label as strings but this was quicker/easier than relabeling)
	df['label'].replace(to_replace = 'pos',value = 4,inplace = True)
	df['label'].replace(to_replace = 'neut',value = 2,inplace = True)
	df['label'].replace(to_replace = 'neg',value = 0,inplace = True)

	df['lema_text'] = (pd.Series(batch_lemmatizer(df['text'],50))) ## preprocess the text
	df = df[df['lema_text'] != '']
	print(df)

	### JUST FOR INSERTING (leave commented) ##
	# db.processed_Training_Data_Three.insert_many(df.to_dict('records'))

	### Single tweet to test preprocessing steps (used in presentation) ###
	# string = '@Sparkywoomy A freshsalad of all fruits &amp; vegetables is a part of the McSparky meald availableee noaw at your nearest McDonalds staore. #trending 😠😠😠 https://t.co/KhWTXElnJb &gt; (a $100 value) :)'
	# batch_lemmatizer([string])





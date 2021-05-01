import math
import symspellpy
import pkg_resources
from symspellpy import SymSpell
from nltk.corpus import words,stopwords
import textblob
import re
import datetime
from sklearn import feature_extraction as sk
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

test_sentence = 'How am   I just finkding thi\'s awegsome templatnbe by @54Mr_Meyer?! Snfag your own copy: https://t.co/DZOQovcLFl @SlidesManiaSM #Free resources for #GoogleSlides &amp; #PowerPoint @test'
# def normalize_tweet(twt):
# 	twt = twt.lower()
# 	twt = emoji_Replacer(twt)
# 	emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',twt) # get emoticons (to add to end important for sentiment removed during further prepro)
# 	twt = re.sub(r'(@[^\s]+(\s|$))|(https?://[^\s]+(\s|$))|(&[^\s]+(\s|$))|[^\w\s\']|\d','',twt)# |(#[^\s]+(\s|$)) remove @ mentions, #, urls, special characters besides ' , and numbers
# 	twt = re.sub(r'\s+',' ',twt)
# 	return (twt)#,emoticons)#twt + ' '.join(emoticons)

def chunker(tweet_list,length,chunksize):
	return(tweet_list[pos:pos + chunksize] for pos in range(0,length,chunksize)) #return a generator for the chunks

def lemmatize_pipe(doc):
	lemma_list = [str(tok.lemma_).lower() for tok in doc if tok.is_alpha] #and tok.text.lower() not in stopwords
	# test spell correction here
	s = ' '.join(lemma_list)
	#print(s)
	return s	
	#sym_spell.lookup_compound(s, max_edit_distance = 2)[0]._term
	#lemma_list

def flatten(L):
	#print(L)
	return [tweet for batch in L for tweet in batch]

def chunk_processor(texts):
	preproc_pipe = []
	process = []
	for twt in texts:
		#print(twt)
		twt = emoji_Replacer(twt)
        #emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',twt) # get emoticons (to add to end important for sentiment removed during further prepro)
		twt = re.sub(r'(@[^\s]+(\s|$))|(#[^\s]+(\s|$))|(https?://[^\s]+(\s|$))|(&[^\s]+(\s|$))|[^\w\s\']|\d','',twt)# |(#[^\s]+(\s|$)) remove @ mentions, #, urls, special characters besides ' , and numbers
		twt = re.sub(r'\s+',' ',twt)
		twt = sym_spell.lookup_compound(twt,max_edit_distance=2)[0]._term
		twt = twt.lower()
		if not twt.isspace():
			process.append(twt)
		#print(twt)
	nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
	for doc in nlp.pipe(process, batch_size=20):
		#print(doc)
		preproc_pipe.append(lemmatize_pipe(doc))
	return preproc_pipe

def batch_lemmatizer(texts,chunksize=100):
	executor = Parallel(n_jobs=4, backend='multiprocessing', prefer="processes")
	do = delayed(chunk_processor)
	tasks = (do(chunk) for chunk in chunker(texts, len(texts), chunksize=chunksize))
	result = executor(tasks)
	return flatten(result)

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


#combine
	#twt = re.sub(r'(@[^\s]+(\s|$))|(#[^\s]+(\s|$))|(https?://[^\s]+(\s|$))','',twt)# remove @ mentions


#stop word removal done in scikit 
	#	#clean_mess = [word for word in rm_tokens if not word.isspace() and word not in stopwords.words('english')]


#symspell to be added if speedup can be done 
	#a = ' '.join(l)
	#r = sym_spell.lookup_compound(a, max_edit_distance = 2)
	#r[0]._term


#this was in something i read not sure why tho so i removed it 
	#twt_list = [ele for ele in twtblob if ele != 'user']
	#print('removed user: '+twt)
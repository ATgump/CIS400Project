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
test_sentence = 'How am   I just finkding thi\'s awegsome templatnbe by @54Mr_Meyer?! Snfag your own copy: https://t.co/DZOQovcLFl @SlidesManiaSM #Free resources for #GoogleSlides &amp; #PowerPoint @test'
def normalize_tweet(twt):
	twt = twt.lower()
	emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',twt) # get emoticons (to add to end important for sentiment removed during further prepro)
	twt = re.sub(r'(@[^\s]+(\s|$))|(#[^\s]+(\s|$))|(https?://[^\s]+(\s|$))|(&[^\s]+(\s|$))|[^\w\s\']|\d','',twt)# remove @ mentions, #, urls, special characters besides ' , and numbers
	twt = re.sub(r'\s+',' ',twt)
	return (twt,emoticons)#twt + ' '.join(emoticons)

def chunker(tweet_list,length,chunksize):
	return(tweet_list[pos:pos + chunksize] for pos in range(0,length,chunksize)) #return a generator for the chunks

def lemmatize_pipe(doc):
	lemma_list = [str(tok.lemma_).lower() for tok in doc if tok.is_alpha] #and tok.text.lower() not in stopwords
	# test spell correction here
	s = ' '.join(lemma_list)
	return sym_spell.lookup_compound(s,max_edit_distance=2)[0]._term.split()	
	#sym_spell.lookup_compound(s, max_edit_distance = 2)[0]._term
	#lemma_list

def flatten(L):
    return [item for S in L for item in S]

def chunk_processor(texts):
	preproc_pipe = []
	nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
	for doc in nlp.pipe(texts, batch_size=20):
		preproc_pipe.append(lemmatize_pipe(doc))
	return preproc_pipe

def batch_lemmatizer(texts,chunksize=100):
	executor = Parallel(n_jobs=4, backend='multiprocessing', prefer="processes")
	do = delayed(chunk_processor)
	tasks = (do(chunk) for chunk in chunker(texts, len(texts), chunksize=chunksize))
	result = executor(tasks)
	return flatten(result)

print(normalize_tweet(test_sentence))

print(sym_spell.lookup_compound(normalize_tweet(test_sentence)[0], max_edit_distance = 2)[0]._term)
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
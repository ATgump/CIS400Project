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

    # tweet_list = [ele for ele in tweet.split() if ele != 'user']
    # clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
    # clean_s = ' '.join(clean_tokens)
    # clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]

#@[A-Za-z0–9]+
import spacy
test_sentence = 'How am   I just finding this awesome template by @54Mr_Meyer?! Snag your own copy: https://t.co/DZOQovcLFl @SlidesManiaSM #Free resources for #GoogleSlides &amp; #PowerPoint @test'
def normalize_tweet(twt):
	twt = twt.lower()
	emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',twt) # get emoticons (to add to end important for sentiment removed during further prepro)
	nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
	twt = re.sub(r'(@[^\s]+(\s|$))|(#[^\s]+(\s|$))|(https?://[^\s]+(\s|$))','',twt)# remove @ mentions, #, and urls 
	twtblob = twt.split()
	rm_tokens = [t for t in twtblob if re.match(r'[^\W\d]*$', t)] # removes any word that has non characters in it includeing ones that dont have a space ... maybe add space in between non chars to split words so copy: -> copy : and doesnt get removed
	sentence = ' '.join(rm_tokens)
	doc = nlp(re.sub(r'\s+',' ',sentence)) #remove extra spaces from sentence then pass it to spacy to lemmatize
	return ' '.join([token.lemma_ for token in doc])+' ' + ' '.join(emoticons)

start_time = datetime.datetime.now()
print(normalize_tweet(test_sentence))
end_time = datetime.datetime.now()
print(end_time-start_time)


def batch_lemmatizer(tweet_collection):
# remove_rt = lambda x: re.sub(‘RT @\w+: ‘,” “,x)
# rt = lambda x: re.sub(“(@[A-Za-z0–9]+)|([⁰-9A-Za-z \t])|(\w+:\/\/\S+)”,” “,x)



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
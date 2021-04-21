import math
import symspellpy
import pkg_resources
from symspellpy import SymSpell
from nltk.corpus import words,stopwords
import textblob
import re

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
	emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',twt) # get emoticons (to add to end important for sentiment removed during further prepro)
	nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
	twt = re.sub(r'\s+',' ',twt) #remove extra white space
	#print('removed white space: '+twt)
	twt = re.sub(r'@[^\s]+(\s|$)','',twt)# remove @ mentions
	#print('removed @ mention: '+twt)
	twt = re.sub(r'#[^\s]+(\s|$)','',twt) #remove #tags
	#print('removed tags: '+twt)
	twt = re.sub(r'https?://[^\s]+(\s|$)','',twt) # remove links
	#print('removed links: '+twt)
	twtblob = twt.split()
	twt_list = [ele for ele in twtblob if ele != 'user']
	#print('removed user: '+twt)
	rm_tokens = [t.lower() for t in twt_list if re.match(r'[^\W\d]*$', t)] # removes any word that has non characters in it includeing ones that dont have a space ... maybe add space in between non chars to split words so copy: -> copy : and doesnt get removed
	#print('removed non chars: '+' '.join(rm_tokens))
	clean_mess = [word for word in rm_tokens if not word.isspace() and word not in stopwords.words('english')]
	#print('removed stop word: '+' '.join(clean_mess))
	#a = ' '.join(l)
	#r = sym_spell.lookup_compound(a, max_edit_distance = 2)
	#r[0]._term
	sentence = ' '.join(textblob.TextBlob(' '.join(clean_mess)).words)
	doc = nlp(sentence)
	return ' '.join([token.lemma_ for token in doc])+' ' + ' '.join(emoticons)


print(normalize_tweet(test_sentence))

# remove_rt = lambda x: re.sub(‘RT @\w+: ‘,” “,x)
# rt = lambda x: re.sub(“(@[A-Za-z0–9]+)|([⁰-9A-Za-z \t])|(\w+:\/\/\S+)”,” “,x)
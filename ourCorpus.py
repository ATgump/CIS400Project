import math
import symspellpy
import pkg_resources
from symspellpy import SymSpell
from nltk.corpus import words



sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

#Corpus class: A collection of text (list of tweets). Calculates TF-IDF of any word in any document in the corpus. 
class corpus:
	def __init__(self,doc_list = [],number_of_docs = 0):
		self.number_of_docs = number_of_docs
		self.doc_list = [self.normalize_tweet(d) for d in doc_list]


	#calculate the TF-IDF of a particular word in a particular document ## Note: should split this into IDF calc then tf after
	def TF_IDF_weight(self,word, doc): 
		num = 0
		#Find the number of documents in the corpus that 'word' exists
		for d in self.doc_list:
			if word in d:
				num = num+1
		try: 
			answ = doc.split().count(word) * math.log(self.number_of_docs/num,2) # TF-IDF forumula applied
			return answ
		except ZeroDivisionError:
			print("This word does not appear anywhere in the corpus")
			return None
	#def TF(self,word, doc):


	def TF_IDF_feature_vector(self): ## return - [{word:TF-IDF value}] where each dict represents a vectorized tweet
		vec = [] 
		for doc in self.doc_list:
			dict1 = dict()
			for word in doc.split():
				dict1[word] = self.TF_IDF_weight(word,doc)
			vec.append(dict1)
		return vec
	
	def add_document(self,doc):
		self.doc_list.append(doc)
		self.number_of_docs = self.number_of_docs + 1


	def normalize_tweet(self,str):
		a = str.strip()
		#remove any special characters and numbers from the string e.g. (!,@,#,4,5,2) 
		l = a.split()
		for w in l:
			if w[0] == '@':
				l.remove(w)
		a = ' '.join(l)
		r = sym_spell.lookup_compound(a, max_edit_distance = 2)
		return r[0]._term


import math

#Corpus class: A collection of text entered by the user (list of documents). Calculates TF-IDF of any word in any document in the corpus. 
class corpus:
	def __init__(self,doc_list = [],number_of_docs = 0):
		self.number_of_docs = number_of_docs
		self.doc_list = doc_list


	#calculate the TF-IDF of a particular word in a particular document ## Note: should split this into IDF calc then tf after
	def TF_IDF_weight(self,word, doc): 
		num = 0
		x = 0
		#Find the number of documents in the corpus that 'word' exists
		while(x != self.number_of_docs):
			if word in self.doc_list[x].words_counted.keys():
				num = num+1
			x = x+1
		try: 
			answ = doc.word_count(word) * math.log(self.number_of_docs/num,2) # TF-IDF forumula applied
		except ZeroDivisionError:
			print("This word does not appear anywhere in the corpus")
		return answ 


	def add_document(self,doc):
		self.doc_list.append(doc)
		self.number_of_docs = self.number_of_docs + 1
	#Turn the corpus into a string. This string lists the documents as well as the words in the documents and includes their weight/word_count 
	
	def __str__(self):
		ww_map = wordWeight_map(self) #Create a dictionary of the weights of words in the corpus
		st = 'Document List: \n\n'
		#Turn the Corpus into *detailed* string
		for doc in self.doc_list:
			categorized = doc.categorize()
			st = st + "Document #" + str(doc.doc_number) + ":\n\tString: " + doc.doc_string + "\n\tWord Count: \n\t\t"
			for (k,v) in doc.words_counted.items():
				st = st + str(k) + ": " + str(v) + " occurrence(s)\n\t\t"
			st = st + "\n\tWord Weight(s): \n\t\t"
			for ((word,doc_number),weight) in ww_map.items():
				if (doc_number == doc.doc_number):
					st = st + str(word) + ":" + str(round(weight,3)) + "\n\t\t"
			st = st + "\n\tWord Frequency Categories:\n\t\tlow: "
			for (word,num) in categorized[0].items():
				st = st + str(word) + ", "
			st = st + "\n\t\tmed: "
			for (word,num) in categorized[1].items():
				st = st + str(word) + ", "
			st = st + "\n\t\thigh: "
			for (word,num) in categorized[2].items():
				st = st + str(word) + ", "
			st = st + "\n"
		return st

#Function to create a dictionary of the form: (word,document #):TF-IDF_weight_value, for every unique word in every document in a supplied corpus
def wordWeight_map(corp):
	weighted_map = dict()
	for doc in corp.doc_list:
		for (word,num) in doc.words_counted.items():
			weighted_map[(word,doc.doc_number)] = corp.TF_IDF_weight(word,doc)
	return weighted_map

#Document class: documents are entered by user (and stored) as strings. The string is also turned into a dictionary (words_counted) of the form (word, occurrences of word in document). Identified by document number
class document:
	def __init__(self,doc_string,words_counted,doc_number):
		self.doc_string = doc_string
		self.words_counted = words_counted
		self.doc_number = doc_number
	def word_count(self,word):
		return self.words_counted[word]
	def __str__(self):
		s = 'String: ' + self.doc_string + '\t Word Count: ' + str(self.words_counted)
		return s
	#Categorize the words in the document into low,med,high frequency categories (using dictionary comprehensions). Return as a tuple
	def categorize(self):
		low = {k:v for (k,v) in self.words_counted.items() if v < 10} 
		med = {k:v for (k,v) in self.words_counted.items() if v >= 10 and v < 20}
		high = {k:v for (k,v) in self.words_counted.items() if v >= 20}
		return (low,med,high)
	#Return a list (using list comprehension) of all the words of length n in the document
	def words_length_n(self,n):
		L = [word for (word,num) in self.words_counted.items() if len(word) == n]
		return L

#Function to turn a string into a dictionary of the form (word,occurences of word in string). Used to create the dictionary stored by document objs. 
def wordCount_map(str):
	a = str.strip()
	#remove any special characters and numbers from the string e.g. (!,@,#,4,5,2) 
	for c in a:
		if not(c.isalpha() or c.isspace()):
			a = a.replace(c,'')
	#ensure that all words are lower case because This == this, remove any double or triple spaces (possible accident in entry), then split into a list of words.
	a = a.lower()
	a = a.replace('  ',' ')
	a = a.replace('   ',' ')
	list1 = a.split()
	d = dict()
	#Iterate through the list of words to count the occurences of the first word in the list, 
	#add, word:# of occurences, to the dictionary. Remove all instances of that word in the list of words. Continue iteration until list is empty.
	while(list1):
		x = list1.count(list1[0])
		s = list1[0]
		d[s] = x
		while(x != 0):
			list1.remove(s)
			x = x-1
	return d

#MAIN#
if __name__ == "__main__":
	corp = corpus()
	doc_num = 0
	#Continue adding user input 'documents' to the corpus until the user has no more documents to enter.
	while(True):
		print('Would you like to add a document to the corpus? (Y/N)')
		response = input()
		if response == 'N' or response == 'n':
			break
		elif response == 'Y' or response == 'y':
			doc_num = doc_num+1
			print('Enter the document you would like to add to the corpus as a string: ')
			s = input()
			d = document(s,wordCount_map(s),doc_num)
			corp.add_document(d)
		else:
			print('Please enter either Y or N\n')
	#Print the corpus detailing all of the documents as well as a TF-IDF weight for every word in every document in the corpus
	print("This is the analysis of the corpus you have created: ")
	print(corp)
	#Have the user enter two numbers, n i, and return a list of the words in document i of length n. Use a try/except to catch any invalid inputs by the user 
	while(True):
		print("create a list of all words of length n in document i, enter in the form of n i ")
		try:
			n,i = [int(x) for x in input().split()]
			print(corp.doc_list[i-1].words_length_n(n))
			break
		except:
			print("This is not a valid entry, ensure that the format is correct and that the document # you have entered is valid")
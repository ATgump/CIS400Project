from twitter import *
from sklearn.naive_bayes import MultinomialNB
import myTwitterCookbook 
import json
from bson import json_util
# Token Authorizations for Jon
CONSUMER_KEY = '7YjYsiv2BFHEayteg6xOlrkII'
CONSUMER_SECRET = 'NI9691P5BzasF7U8mGDtuzI4RxE67NJ8tHUzDNG74o8aiMk7VD'
OAUTH_TOKEN = '1415807012-z9R1lUy19txxE6nZbHjrP5KHelO6vRZp84YrMQd'
OAUTH_TOKEN_SECRET = 'WNJMBcoyTwRNx30e1QRKzVnl0TXXW5cP0sjxQtEPSrugi'


# NOTE: The seach api is fairly confusing in the results will probably need to ask prof about it OR operator is
# limiting the number of tweets found which doesnt make any sense (may need to just have repeated requests with simpler queries)



#mongo connection
# client = pymongo.MongoClient('mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true')

# Twitter API call from CookBook and Class Slides
# t = Twitter(auth=(OAUTH_TOKEN, OAUTH_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET))
twitter_api = myTwitterCookbook.oauth_login() #Avery's connection
#len of q = 494
q = "'Five Guys' OR Marios OR Sylvias OR Defontes OR Patsys OR Barbetta OR Keens OR Katzs OR Bamontes OR Frenchette OR 'The Modern' OR Cookshop OR 'Casa Lever' OR 'Mercer Kitchen' OR \
'Charlie Bird' OR 'Le Bernardin' OR Tong OR 'Runaway Roof' OR 'La Grenouille' OR \
'P.J. Clarkes' OR Soothr OR Raouls OR Forsynthia OR 'Wo Hop' OR Adda OR 'Crown Shy' OR Mokyo OR Claro OR 'Via Carota' OR Atoboy \
OR 'Usha Foods' OR 'Peter Luger' OR 'The Rainbow Room' OR Totonnos OR Delmonicos OR Raos OR 'Nathans Famous' \
OR 'Johns Pizzeria'"


#test = '"Five%20Guys"%20OR%20Marios%20OR%20Sylvias%20OR%20Defontes%20OR%20Patsys%20OR%20Barbetta%20OR%20Keens%20OR%20Katzs%20OR%20Bamontes%20OR%20Frenchette%20OR%20Cookshop'
#q3 = "%27Five%20Guys%27"
#q2 = 'McDonalds'# OR "Five Guys"'

results = myTwitterCookbook.twitter_search(twitter_api, q2, max_results=2000)

# lang = 'en'
print(len(results))        
mConnection = 'mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true'

#ids = myTwitterCookbook.save_to_mongo(results, 'test_Search_Mining_DB', 'Train Data', host=mConnection)

#col = myTwitterCookbook.load_from_mongo('search_results', q, host=mConnection)

#print data base info 
print(len(col))
print("\n\n\n")
#for tweet in col:
 #   print(json_util.dumps(tweet,indent =1))


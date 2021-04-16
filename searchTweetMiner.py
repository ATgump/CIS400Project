from twitter import *
from sklearn.naive_bayes import MultinomialNB
import myTwitterCookbook 
import json

# Token Authorizations for Jon
CONSUMER_KEY = '7YjYsiv2BFHEayteg6xOlrkII'
CONSUMER_SECRET = 'NI9691P5BzasF7U8mGDtuzI4RxE67NJ8tHUzDNG74o8aiMk7VD'
OAUTH_TOKEN = '1415807012-z9R1lUy19txxE6nZbHjrP5KHelO6vRZp84YrMQd'
OAUTH_TOKEN_SECRET = 'WNJMBcoyTwRNx30e1QRKzVnl0TXXW5cP0sjxQtEPSrugi'

# Twitter API call from CookBook and Class Slides
# t = Twitter(auth=(OAUTH_TOKEN, OAUTH_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET))
twitter_api = myTwitterCookbook.oauth_login() 

# Function to collect tweets
# q = "FiveGuys"
# probe1 = myTwitterCookbook.twitter_search(t, q, 10)
# print(json.dumps(probe1[0], indent=1))

# twitter_api = oauth_login()

q = "Five Guys"
results = myTwitterCookbook.twitter_search(twitter_api, q, max_results=10)
        
# Show one sample search result by slicing the list...
print(json.dumps(results[0], indent=1))

#mongo connection
# client = pymongo.MongoClient('mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true')
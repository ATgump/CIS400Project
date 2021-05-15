import tweepy
import pymongo
from urllib3.exceptions import ProtocolError

## My Twitter API key -- mongo connection string for CIS400 Project User -- tweet_DB## 
CONSUMER_KEY = '4B21fng1De6ivIbzQ44ZyhtXm'
CONSUMER_SECRET = 'T8wmdnY9TYNvihFaZduNlRoFOxT5is2B43O1va6k0XGiYfo8C4'
OATH_TOKEN = '1359226761515593731-zJjwg8wZjFhvWgNWsOx7sBvbhBacGM'
OATH_TOKEN_SECRET = 'RtmaAQoocQnV7v3KJetxlekdnJIGee7JYyr1doykNGzii'
mConnection = 'mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true'
client = pymongo.MongoClient(mConnection)
db = client['tweet_DB']

## Get oath connection 
def connect_to_twitter_OAuth():
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OATH_TOKEN, OATH_TOKEN_SECRET)
    api = tweepy.API(auth)
    return api

## Class for tweepy stream listener
class StreamListener(tweepy.StreamListener):
    tweet_counter = 0
    def __init__(self, api=None):
        super(StreamListener, self).__init__()
        self.tweet_counter = 0
    def on_status(self, status):   
        if self.tweet_counter < 100000: ## Tweet collection limit
            if hasattr(status,'retweeted_status'): ## remove retweets
                return
            self.tweet_counter = self.tweet_counter+1

            ## Select Fields to add to Mongo ## 
            db_add = {}
            db_add['_id'] = status._json['id']
            db_add['lang'] = status._json['lang']
            db_add['id_str'] = status._json['id_str']
            if not status.truncated:
                db_add['text'] = status.text
            else:
                db_add['text'] = status.extended_tweet['full_text']
            db_add['entities'] = status._json['entities']
            
            ## Determine the tweets Topic Restuarant and add it to the
            ## Correct database
            mcd = {'McDonalds','mcDonalds','mcdonalds','@McDonalds','mcdonald'}
            wend = {'Wendys','wendys','wendy','@Wendys'}
            bk = {'Burger King','burger King','burger king','@BurgerKing','@burgerking','@Burgerking','@burgerKing','BurgerKing','burgerking','Burgerking'}
            try:

                ## Add McD tweet to McD DB (experimental_Data_McDonalds)
                if any(x in db_add['text'] for x in mcd):
                    #db.experimental_Data_McDonalds.insert_one(db_add)
                    print('# of tweets collected:: '+str(self.tweet_counter))
                    print("Tweet added to McDonalds DB: " + db_add['text'])
                    return

                ## Add Wendys tweet to Wendys DB (experimental_Data_Wendys)   
                elif any(x in db_add['text'] for x in wend):
                    #db.experimental_Data_Wendys.insert_one(db_add)
                    print('# of tweets collected:: '+str(self.tweet_counter))
                    print("Tweet added to Wendys DB: " + db_add['text'])
                    return

                ## Add BK tweet to BK DB (experimental_Data_BK)    
                elif any(x in db_add['text'] for x in bk):
                    #db.experimental_Data_BK.insert_one(db_add)
                    print('# of tweets collected:: '+str(self.tweet_counter))
                    print("Tweet added to BurgerKing DB: " + db_add['text'])
                    return
                
                ## If the topic cannot be determined don't add it to db
                else:
                    print("This tweet was not found to be about a restraunt in the query")
            
            except:
                print("exception occured")
                return
        else:
            print("FINISHED COLLECTION")
            return False

    ## Exception Handling        
    def on_error(self, status_code):
        if status_code == 420:
            print("------------------LIMITED-----------")
            return False

## Open a stream using Twitters streaming API
twitter_api = connect_to_twitter_OAuth()
myStreamListener = StreamListener()
myStream = tweepy.Stream(auth = twitter_api.auth, listener=myStreamListener, tweet_mode = 'extended')

## Stream/ More exception handling
while True:
    q4 = ["McDonalds","Wendys",'Burger King']
    try:
        myStream.filter(track=q4, languages=['en'])
    except(ProtocolError,AttributeError):
        print("encountered a protocol error")
        continue
    except:
        print("encountered a protocol error")
        continue


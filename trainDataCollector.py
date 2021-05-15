import tweepy
import pymongo

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

            ## Save to mongo/print tweet ## 
            #db.unprocessed_Tweets.insert_one(db_add)
            
            print('# of tweets collected:: '+str(self.tweet_counter))
            print(db_add['text'])
        else:
            print("FINISHED COLLECTION")
            return False
    
    ## Error Handling ## 
    def on_error(self, status_code):
        if status_code == 420:
            print("------------------LIMITED-----------")
            return False

## Connect to OAuth and Open stream listener ##
twitter_api = connect_to_twitter_OAuth()
myStreamListener = StreamListener()
myStream = tweepy.Stream(auth = twitter_api.auth, listener=myStreamListener, tweet_mode = 'extended')

## Filter english tweets with q4 for query terms ##
q4 = ["McDonalds","Wendys",'Burger King','Pizza Hut',"Whataburger","In-N-OutBurger","White Castle","Starbucks","Auntie Anne\'s","Popeyes","Chick-fil-A",'Taco Bell',"Arby\'s","Dairy Queen"]
myStream.filter(track=q4, languages=['en'])
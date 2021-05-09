import tweepy
import myTwitterCookbook
import json
import pymongo
from urllib3.exceptions import ProtocolError
CONSUMER_KEY = '4B21fng1De6ivIbzQ44ZyhtXm'
CONSUMER_SECRET = 'T8wmdnY9TYNvihFaZduNlRoFOxT5is2B43O1va6k0XGiYfo8C4'
OATH_TOKEN = '1359226761515593731-zJjwg8wZjFhvWgNWsOx7sBvbhBacGM'
OATH_TOKEN_SECRET = 'RtmaAQoocQnV7v3KJetxlekdnJIGee7JYyr1doykNGzii'
mConnection = 'mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true'
client = pymongo.MongoClient(mConnection)
db = client['tweet_DB']
def connect_to_twitter_OAuth():
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OATH_TOKEN, OATH_TOKEN_SECRET)
    api = tweepy.API(auth)
    return api

class StreamListener(tweepy.StreamListener):
    tweet_counter = 0
    def __init__(self, api=None):
        super(StreamListener, self).__init__()
        self.tweet_counter = 0
    def on_status(self, status):   
        if self.tweet_counter < 100000:
            if hasattr(status,'retweeted_status'):
                return
            self.tweet_counter = self.tweet_counter+1
            db_add = {}
            db_add['_id'] = status._json['id']
            db_add['lang'] = status._json['lang']
            db_add['id_str'] = status._json['id_str']
            if not status.truncated:
                db_add['text'] = status.text
            else:
                db_add['text'] = status.extended_tweet['full_text']
            db_add['entities'] = status._json['entities']
            
            mcd = {'McDonalds','mcDonalds','mcdonalds','@McDonalds','mcdonald'}
            wend = {'Wendys','wendys','wendy','@Wendys'}
            bk = {'Burger King','burger King','burger king','@BurgerKing','@burgerking','@Burgerking','@burgerKing','BurgerKing','burgerking','Burgerking'}
            try:
                if any(x in db_add['text'] for x in mcd):
                    db.experimental_Data_McDonalds.insert_one(db_add)
                    print('# of tweets collected:: '+str(self.tweet_counter))
                    print("Tweet added to McDonalds DB: " + db_add['text'])
                    return
                elif any(x in db_add['text'] for x in wend):
                    db.experimental_Data_Wendys.insert_one(db_add)
                    print('# of tweets collected:: '+str(self.tweet_counter))
                    print("Tweet added to Wendys DB: " + db_add['text'])
                    return
                elif any(x in db_add['text'] for x in bk):
                    db.experimental_Data_BK.insert_one(db_add)
                    print('# of tweets collected:: '+str(self.tweet_counter))
                    print("Tweet added to BurgerKing DB: " + db_add['text'])
                    return
                else:
                    print("This tweet was not found to be about a restraunt in the query")
            
            except:
                print("UH oh an exception occured")
                return
            # print('# of tweets collected:: '+str(self.tweet_counter))
            # print(db_add['text'])
        else:
            print("FINISHED COLLECTION")
            return False
    def on_error(self, status_code):
        if status_code == 420:
            print("------------------LIMITED-----------")
            return False
tweet_list = []
twitter_api = connect_to_twitter_OAuth()
myStreamListener = StreamListener()
myStream = tweepy.Stream(auth = twitter_api.auth, listener=myStreamListener, tweet_mode = 'extended')
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


#maybe remove this:::: Gruesome factory farms are pumping animals full of antibiotics, helping to create vicious new superbugs -- but now @McDonalds has a golden opportunity to help stop the next pandemic! Add your voice!
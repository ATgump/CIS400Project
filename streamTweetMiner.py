import twitter
import sys
import json
from myTwitterCookbook import oauth_login, make_twitter_request
import myTwitterCookbook

# Returns an instance of twitter.Twitter
twitter_api = oauth_login()

# Reference the self.auth parameter
twitter_stream = twitter.TwitterStream(auth=twitter_api.auth)

# Query terms
q = "covid,'COVID-19'"

#removed : claro
q2 = "'Five Guys',Marios,Sylvias,Defontes,Patsys,Barbetta,Keens,Katzs,\
Bamontes,Frenchette,'The Modern',Cookshop,'Casa Lever','Mercer Kitchen',\
'Charlie Bird','Le Bernardin',Tong,'Runaway Roof','La Grenouille',\
'P.J. Clarkes',Soothr,Raouls,Forsynthia,'Wo Hop',Adda,'Crown Shy',\
Mokyo,'Via Carota',Atoboy,'Usha Foods','Peter Luger','The Rainbow Room',\
Totonnos,Delmonicos,Raos,'Nathans Famous','Johns Pizzeria'"

q3 = "McDonalds,Wendys,'Burger King',Subway,'Pizza Hut'"

print('Filtering the public timeline for track={0}'.format(q3), file=sys.stderr)
sys.stderr.flush()

# See https://developer.twitter.com/en/docs/tutorials/consuming-streaming-data
# stream = twitter_stream.statuses.filter(track=q)
#stream = make_twitter_request(twitter_stream.statuses.filter, track=q)
#stream = make_twitter_request(twitter_stream.statuses.filter, track=q, delimited='length', stallings = True)
stream = make_twitter_request(twitter_stream.statuses.filter, track=q3, stallings = True)

# For illustrative purposes, when all else fails, search for Justin Bieber
# and something is sure to turn up (at least, on Twitter)

tweet_list = []



mConnection = 'mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true'
# box for nyc : -74,40,-73,41
i = 0

for tweet in stream:
    try:
        if len(tweet_list) == 10:
            print("restart the count")
            ids = myTwitterCookbook.save_to_mongo(tweet_list, 'test_Search_Mining_DB', 'T', host=mConnection)
            i = i+1
            tweet_list = []   
        elif i == 10:
            print("broke")
            break
        elif not(type(tweet) is int):
            if not(tweet.get('retweeted_status',False)) and tweet['lang'] == 'en':
                print("appended: " + tweet['text'][:10])
                db_add = {}
                db_add['_id'] = tweet['id']
                db_add['id_str'] = tweet['id_str']
                db_add['text'] = tweet['extended_tweet']['full_text']
                db_add['entities'] = tweet['entities']
                tweet_list.append(db_add)   
        print (tweet['text']) 
    except:
        #print("passed")
        pass

import twitter 
import sys
from sys import maxsize as maxint
import time
from urllib.error import URLError
from http.client import BadStatusLine
import json
from functools import partial
import networkx

#Get an authenticator obj to use with twitter API calls:: taken from twitter cookbook

def oauth_login():
	CONSUMER_KEY = '4B21fng1De6ivIbzQ44ZyhtXm'
	CONSUMER_SECRET = 'T8wmdnY9TYNvihFaZduNlRoFOxT5is2B43O1va6k0XGiYfo8C4'
	OATH_TOKEN = '1359226761515593731-zJjwg8wZjFhvWgNWsOx7sBvbhBacGM'
	OATH_TOKEN_SECRET = 'RtmaAQoocQnV7v3KJetxlekdnJIGee7JYyr1doykNGzii'
	auth = twitter.oauth.OAuth(OATH_TOKEN,OATH_TOKEN_SECRET,CONSUMER_KEY,CONSUMER_SECRET)
	twitter_api_authenticated = twitter.Twitter(auth= auth)
	return twitter_api_authenticated

#Handle twitter exceptions so program doesn't crash when twitter throws errors at you:: taken from twitter cookbook

def make_twitter_request(twitter_api_func, max_errors=10, *args, **kw): 
    
    # A nested helper function that handles common HTTPErrors. Return an updated
    # value for wait_period if the problem is a 500 level error. Block until the
    # rate limit is reset if it's a rate limiting issue (429 error). Returns None
    # for 401 and 404 errors, which requires special handling by the caller.
    def handle_twitter_http_error(e, wait_period=2, sleep_when_rate_limited=True):
    
        if wait_period > 3600: # Seconds
            print('Too many retries. Quitting.', file=sys.stderr)
            raise e
    
        # See https://developer.twitter.com/en/docs/basics/response-codes
        # for common codes
    
        if e.e.code == 401:
            print('Encountered 401 Error (Not Authorized)', file=sys.stderr)
            return None
        elif e.e.code == 404:
            print('Encountered 404 Error (Not Found)', file=sys.stderr)
            return None
        elif e.e.code == 429: 
            print('Encountered 429 Error (Rate Limit Exceeded)', file=sys.stderr)
            if sleep_when_rate_limited:
                print("Retrying in 15 minutes...ZzZ...", file=sys.stderr)
                sys.stderr.flush()
                time.sleep(60*15 + 5)
                print('...ZzZ...Awake now and trying again.', file=sys.stderr)
                return 2
            else:
                raise e # Caller must handle the rate limiting issue
        elif e.e.code in (500, 502, 503, 504):
            print('Encountered {0} Error. Retrying in {1} seconds'                  .format(e.e.code, wait_period), file=sys.stderr)
            time.sleep(wait_period)
            wait_period *= 1.5
            return wait_period
        else:
            raise e

    # End of nested helper function
    
    wait_period = 2 
    error_count = 0 

    while True:
        try:
            return twitter_api_func(*args, **kw)
        except twitter.api.TwitterHTTPError as e:
            error_count = 0 
            wait_period = handle_twitter_http_error(e, wait_period)
            if wait_period is None:
                return
        except URLError as e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print("URLError encountered. Continuing.", file=sys.stderr)
            if error_count > max_errors:
                print("Too many consecutive errors...bailing out.", file=sys.stderr)
                raise
        except BadStatusLine as e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print("BadStatusLine encountered. Continuing.", file=sys.stderr)
            if error_count > max_errors:
                print("Too many consecutive errors...bailing out.", file=sys.stderr)
                raise

# Input: screen_name or user_ID. Return: The list of friends,the list of followers of the input id/screen_name :: Taken from twitter cookbook

def get_friends_followers_ids(twitter_api, screen_name=None, user_id=None,
                              friends_limit=maxint, followers_limit=maxint):
    
    # Must have either screen_name or user_id (logical xor)
    assert (screen_name != None) != (user_id != None),     "Must have screen_name or user_id, but not both"
    
    # See http://bit.ly/2GcjKJP and http://bit.ly/2rFz90N for details
    # on API parameters
    
    get_friends_ids = partial(make_twitter_request, twitter_api.friends.ids, 
                              count=5000)
    get_followers_ids = partial(make_twitter_request, twitter_api.followers.ids, 
                                count=5000)

    friends_ids, followers_ids = [], []
    
    for twitter_api_func, limit, ids, label in [
                    [get_friends_ids, friends_limit, friends_ids, "friends"], 
                    [get_followers_ids, followers_limit, followers_ids, "followers"]
                ]:
        
        if limit == 0: continue
        
        cursor = -1
        while cursor != 0:
        
            # Use make_twitter_request via the partially bound callable...
            if screen_name: 
                response = twitter_api_func(screen_name=screen_name, cursor=cursor)
            else: # user_id
                response = twitter_api_func(user_id=user_id, cursor=cursor)

            if response is not None:
                ids += response['ids']
                cursor = response['next_cursor']
        
            print('Fetched {0} total {1} ids for {2}'.format(len(ids),                  label, (user_id or screen_name)),file=sys.stderr)
        
            # XXX: You may want to store data during each iteration to provide an 
            # an additional layer of protection from exceptional circumstances
        
            if len(ids) >= limit or response is None:
                break

    # Do something useful with the IDs, like store them to disk...
    return friends_ids[:friends_limit], followers_ids[:followers_limit]


# input: list of screen_names or user_ids. Return: a dictionary of form {(user_id/screenname:User_object_json_format)} that contains user infomation for every user in the input list :: taken from twitter cookbook

def get_user_profile(twitter_api, screen_names=None, user_ids=None):
   
    # Must have either screen_name or user_id (logical xor)
    assert (screen_names != None) != (user_ids != None),     "Must have screen_names or user_ids, but not both"
    
    items_to_info = {}

    items = screen_names or user_ids
    
    while len(items) > 0:

        # Process 100 items at a time per the API specifications for /users/lookup.
        # See http://bit.ly/2Gcjfzr for details.
        
        items_str = ','.join([str(item) for item in items[:100]])
        items = items[100:]

        if screen_names:
            response = make_twitter_request(twitter_api.users.lookup, 
                                            screen_name=items_str)
        else: # user_ids
            response = make_twitter_request(twitter_api.users.lookup, 
                                            user_id=items_str)
    
        for user_info in response:
            if screen_names:
                items_to_info[user_info['screen_name']] = user_info
            else: # user_ids
                items_to_info[user_info['id']] = user_info

    return items_to_info

# input: screen_name - a starting point screen name to crawl from, limit - the limit of friends/followers collected for calculating reciprocal users. 
# Return: a graph generated from the nodes collected while crawling. 
# Adapted from twitter cookbook but primarily my own code

def my_crawler_A2(twitter_api, screen_name, limit=5000):

    # Create an empty graph. Get the user_id for the input screen_name for consitancy (will work with IDs). 

     my_Graph = networkx.Graph()
     seed_id = str(twitter_api.users.show(screen_name=screen_name)['id'])

     # Get the top 5 reciprocal friends for screen_name, add the nodes/edges for these users to the graph, and put them in the queue (first batch for the crawler)
  
     friends, followers = get_friends_followers_ids(twitter_api, user_id=seed_id,friends_limit=limit, followers_limit=limit)
     reciprocal = list(set.intersection(set(friends),set(followers)))                                        # calculate reciprocal friends from user_id's list of friends/followers
     recip_users = get_user_profile(twitter_api,user_ids=reciprocal)                                         # get user profiles for these friends to sort by follower count
     recip_users = sorted(recip_users,key = lambda x:recip_users.get(x)["followers_count"], reverse = True)  # get a list of User_ID sorted by follower count in descending order
     next_queue = recip_users[:5]                                                                            # add the first batch of reciprocal users to the (initial) queue
     my_Graph.add_edges_from([(seed_id,y) for y in next_queue])                                              # add initial nodes to graph
     queue = []
    
     # Crawl twitter until at least 100 nodes have been added to the graph.

     while my_Graph.number_of_nodes() < 100:

         # queue changing for continuing breadth first search
         (queue,next_queue) = (next_queue,[])
         
         # for every id in the queue: calculate top 5 reciprocal friends, add (if possible) 5 unique nodes (users) to the graph from these reciprocal friends and connect nodes with edges as needed. 
         # Keep adding reciprocal friend id's to the next_queue until the number of nodes in the graph + maximum possible number of nodes added to the graph in next cycle of crawl >= 100. I.E. limit the next queue based off the anticipated # of nodes generated during the next cycle of crawling

         for rid in queue:
             
             # Calculate list of reciprocal friends sorted by follower count in descending order

             friend_ids, follower_ids = get_friends_followers_ids(twitter_api, user_id=rid,friends_limit=limit, followers_limit=limit)
             reciprocal = list(set.intersection(set(friend_ids),set(follower_ids)))                                 # calculate reciprocal friends from user_id's list of friends/followers
             recip_users = get_user_profile(twitter_api,user_ids=reciprocal)                                        # get user profiles for these friends to sort by follower count
             recip_users = sorted(recip_users,key = lambda x:recip_users.get(x)["followers_count"], reverse = True) # get a list of User_ID sorted by follower count in descending order

             # Add top 5 of the users to the graph (unique), if a user is already added to the graph, draw the appropriate edge in the graph and add a different user. 

             top_five_users = []
             for user_ID in recip_users:
                if len(top_five_users) == 5:
                    break
                elif user_ID in my_Graph.nodes():
                    my_Graph.add_edge(rid,user_ID)
                else:
                    top_five_users.append(user_ID)
              
             # add the appropriate edges between the reciprocal friends
             my_Graph.add_edges_from([(rid,y) for y in top_five_users])       

             # limit the next queue based off the anticipated # of nodes generated during the next cycle of crawling.   
             if (my_Graph.number_of_nodes() + len(next_queue)*5) < 100:  
                next_queue += top_five_users

     return my_Graph


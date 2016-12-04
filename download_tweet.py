import os
import csv
import pandas
import tweepy
from tweepy import OAuthHandler
import ipdb
def authenticate():
    CONSUMER_KEY ='jjxmqFnVhaVT0HNK42eeM06Ha'
    CONSUMER_SECRET='XUrZ0PxN2WxB6p0eZC1VWX0tnwN1i5nfS0xcGGZzLwda6nBB3U'
    ACCESS_KEY = '804687676775370753-NfkgmdPH40mASb7QfXfGoRZy7wIqBsK'
    ACCESS_SECRET = 'qAbW37b08J7SnbP9EohvHJ8YyhC3EMlKZbX61PJwcdIuy'
    auth = OAuthHandler(CONSUMER_KEY,CONSUMER_SECRET)
    api = tweepy.API(auth)
    auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
    #search
    api = tweepy.API(auth)
    return api
def get_all_tweets(screen_name, file_path, api):
    #initialize a list to hold all the tweepy Tweets
    fout = open(file_path, 'wb')
    writer = csv.writer(fout)
    writer.writerow(["id","created_at","text"])
    print ('get tweet of user %s' % screen_name) 
    #make initial request for most recent tweets (200 is the maximum allowed count)
    try:
        new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    except Exception as e:
        print ('Error %s, message: %s' % (screen_name, e.message))
        fout.close()
        return 

    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode('ascii', errors='ignore')] for tweet in new_tweets]
        writer.writerows(outtweets)
        
        #update the id of the oldest tweet less one
        oldest = new_tweets[-1].id - 1
        #all subsiquent requests use the max_id param to prevent duplicates
        try:
            new_tweets = api.user_timeline(screen_name=screen_name,count=200,max_id=oldest)
        except Exception as e:
            print ('Error %s, message: %s' % (screen_name, e.message))
            break
    
    fout.close()
       
def get_user_tweets():
    api = authenticate()
    df_user_id = pandas.read_csv('userid_labels.csv',dtype=str)
    missing_user = []
    for user_id in df_user_id.sns_id:
        try:
            user_profile = api.lookup_users(user_ids=[user_id])
            file_path = 'tweet_ascii/%s_%s.csv' % (user_profile[0].screen_name, user_id)
            if not os.path.isfile(file_path):
                get_all_tweets(user_profile[0].screen_name, file_path, api)
        except Exception as e:
            print ('Error %s, message: %s' % (user_id, e.message))
            missing_user.append(user_id)
    with open('missing_user.txt', wt) as f:
        f.write('\n'.join(missing_user))
#        print user.name
#        print user.description
#        print user.followers_count
#        print user.statuses_count
#        print user.url
if __name__ == '__main__':
    get_user_tweets()


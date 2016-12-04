import os
import csv
import pandas
import ipdb
from twython import Twython # pip install twython
import time # standard lib
def authenticate():
    CONSUMER_KEY='jjxmqFnVhaVT0HNK42eeM06Ha'
    CONSUMER_SECRET='XUrZ0PxN2WxB6p0eZC1VWX0tnwN1i5nfS0xcGGZzLwda6nBB3U'
    ACCESS_KEY = '804687676775370753-NfkgmdPH40mASb7QfXfGoRZy7wIqBsK'
    ACCESS_SECRET = 'qAbW37b08J7SnbP9EohvHJ8YyhC3EMlKZbX61PJwcdIuy'
    twitter = Twython(CONSUMER_KEY,CONSUMER_SECRET,ACCESS_KEY,ACCESS_SECRET)
    return twitter

def get_all_tweets(user_id, screen_name, twitter):
    fout = open('tweet_v2/%s.txt' % user_id, 'wt')

    lis = [467020906049835008] ## this is the latest starting tweet id
    try:
        for i in range(0, 16): ## iterate through all tweets
            ## tweet extract method with the last list item as the max_id
            user_timeline = twitter.get_user_timeline(screen_name=screen_name,
                count=200, include_retweets=False, max_id=lis[-1])
            #time.sleep(300) ## 5 minute rest between api calls
            for tweet in user_timeline:
                fout.write(tweet['text'].encode('utf-8') + '\n') 
                lis.append(tweet['id']) ## append tweet id's
    except Exception as e:
        print ('Error ' + screen_name)

    fout.close()
def get_user_tweets():
    twitter = authenticate()
    df_user_id = pandas.read_csv('user_id_ls.csv',dtype=str)
    for user in df_user_id.sns_id:
        user_profile =twitter.lookup_user(user_id=[user])
        if not os.path.isfile('tweet_v2/%s.txt' % user):
            print ('Get tweets of user %s' % user)
            get_all_tweets(user, user_profile[0]['screen_name'], twitter)
#        print user.name
#        print user.description
#        print user.followers_count
#        print user.statuses_count
#        print user.url
if __name__ == '__main__':
    get_user_tweets()


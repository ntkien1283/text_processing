import  pickle
import os
import string
import re
#from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from optparse import OptionParser
import preprocessor
import pandas
import ipdb
import time # standard lib
import tweepy
from tweepy import OAuthHandler

#http://stevenloria.com/how-to-build-a-text-classification-system-with-python-and-textblob/
class UserAnalysis:
    def __init__(self, home_dir, tweet_dir):
        self.home_dir = home_dir
        self.tweet_dir = tweet_dir
        self.corpus = []
        self.labels = []
    def get_tweet_api(self):
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
    def preprocess_text(self, text):
        text = preprocessor.clean(text).lower()
#        stop_words = set(stopwords.words('english'))
#        word_tokens = word_tokenize(text)
#        filtered_sentence = [w for w in word_tokens if not (w in stop_words or w in string.punctuation)]
        punctuation = string.punctuation
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        text = regex.sub('', text)
        return text
    def update_corpus(self, user_id, screen_name):
        file_path = os.path.join(self.tweet_dir, '%s_%s.csv' % (screen_name, user_id))
        df_tweet = pandas.read_csv(file_path, dtype=str)
        if df_tweet.shape[0] == 0:
            return False
        user_tweets = ' '.join(df_tweet.text[df_tweet.text.notnull()].tolist())
        user_tweets = self.preprocess_text(user_tweets)
        self.corpus.append(user_tweets)
        return True
    def create_feature_matrix(self):
        tfidf = TfidfVectorizer(min_df=1)
        feature_mat = tfidf.fit_transform(self.corpus)
        pickle.dump(feature_mat, open(os.path.join(self.home_dir, 'feature_mat.pickle'), 'wb'))
    def analyze(self):
        user_labels = pandas.read_csv(os.path.join(self.home_dir, 'userid_labels_screenname.csv'), dtype=str)
        user_labels = user_labels.loc[user_labels.screen_name.notnull(), :]
        user_labels['has_tweet'] = True
        for ind, row in user_labels.loc[:5,:].iterrows():
            user_labels.loc[ind, 'has_tweet'] = self.update_corpus(row.sns_id, row.screen_name)
        ipdb.set_trace()
        self.create_feature_matrix()
if __name__ == '__main__':
    optparser = OptionParser()
    optparser.add_option('-p', 
                         dest='home_dir')
                         
    optparser.add_option('-t', 
                         dest='tweet_dir')
 

    (options, args) = optparser.parse_args()
#    corpus = ['There is a table', 'There is a chair', 'There is a dog']
#    counter = CountVectorizer(min_df=1)
#    ipdb.set_trace()
#    tf_transformer = TfidfTransformer(smooth_idf=False)
#    count_res = counter.fit_transform(corpus)
#    tf_weight = tf_transformer.fit_transform(count_res)
#    tf_vector = TfidfVectorizer(min_df=1)
#    tf_weight1 = tf_vector.fit_transform(corpus)
    user = UserAnalysis(options.home_dir, options.tweet_dir)
    user.analyze()


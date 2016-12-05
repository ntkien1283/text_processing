import numpy
import  pickle
import os
import string
import re
#from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn import metrics
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from optparse import OptionParser
import preprocessor
import pandas
import ipdb
import time # standard lib
import tweepy
from tweepy import OAuthHandler

#http://stevenloria.com/how-to-build-a-text-classification-system-with-python-and-textblob/
#http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
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
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        text = regex.sub('', text)
        return text
    def create_feature_matrix(self):
        tfidf = TfidfVectorizer(min_df=1)
        feature_mat = tfidf.fit_transform(self.corpus)
        pickle.dump(tfidf, open(os.path.join(self.home_dir, 'tfidf.pickle'), 'wb'))
        pickle.dump(feature_mat, open(os.path.join(self.home_dir, 'feature_mat.pickle'), 'wb'))
        pickle.dump(self.final_label, open(os.path.join(self.home_dir, 'final_label.pickle'), 'wb'))
    def analyze(self):
        user_labels = pandas.read_csv(os.path.join(self.home_dir, 'userid_labels_screenname.csv'), dtype=str)
        user_labels = user_labels.loc[user_labels.screen_name.notnull(), :]
        user_labels['has_tweet'] = True
        for ind, row in user_labels.iterrows():
            file_path = os.path.join(self.tweet_dir, '%s_%s.csv' % (row.screen_name, row.sns_id))
            df_tweet = pandas.read_csv(file_path, dtype=str)
            if df_tweet.shape[0] == 0:
                user_labels.loc[ind, 'has_tweet'] = False
            else:
                user_tweets = ' '.join(df_tweet.text[df_tweet.text.notnull()].tolist())
                user_tweets = self.preprocess_text(user_tweets)
                self.corpus.append(user_tweets)

        self.final_label = user_labels.loc[user_labels.has_tweet==True, 'Category'].tolist()
        self.create_feature_matrix()
    def prediction_analysis(self, model_selection):
        tfidf = pickle.load(open(os.path.join(self.home_dir, 'tfidf.pickle'), 'rb'))
        feature_mat = pickle.load(open(os.path.join(self.home_dir, 'feature_mat.pickle'), 'rb'))
        labels = pickle.load(open(os.path.join(self.home_dir, 'final_label.pickle'), 'rb'))
        labels = [1 if i == 'Politician' else 2 if i == 'Trader' else 3 for i in labels]
        X_train, X_test, y_train, y_test = train_test_split(feature_mat, labels, test_size=0.3, random_state=0)
        if model_selection:  
            #Model selection on the training set
            models = dict()
            models['Naive_Bayes'] =  MultinomialNB()
            models['SVM'] = SVC(kernel='linear', C=1)
            models['Decision_Tree'] = tree.DecisionTreeClassifier()
            #Using feature selection by idf
            sorted_ind = numpy.argsort(-tfidf.idf_)
            #Keep top k% of the features
            sel_feature_portion = [0.1, 0.3, 0.5, 0.9, 1]
            for portion in sel_feature_portion:
                remove_feature_ind = sorted_ind[int(portion*len(sorted_ind)):]
                for method_name, algo in models.items():
                    sel_features = numpy.delete(X_train.toarray(), remove_feature_ind, axis=1)
                    predicted = cross_val_predict(algo, sel_features, y_train, cv=5)
                    result = metrics.accuracy_score(predicted, y_train)
                    print ('Select top %.2f portion of features, %s: %.2f accuracy' % (portion, method_name, result))
        else:
             
            #Apply on the test set
if __name__ == '__main__':
    optparser = OptionParser()
    optparser.add_option('-p', 
                         dest='home_dir')
                         
    optparser.add_option('-t', 
                         dest='tweet_dir')
 

    (options, args) = optparser.parse_args()
    user = UserAnalysis(options.home_dir, options.tweet_dir)
    user.prediction_analysis()


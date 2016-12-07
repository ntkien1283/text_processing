import enchant
import csv
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
from sklearn.ensemble import RandomForestClassifier
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
        #Improvement: These keys should not be  included in the code but rather stored in local machine only
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

    def get_user_tweets(self, screen_name, file_path, api):
        #initialize a list to hold all the tweepy Tweets
        fout = open(file_path, 'wt')
        writer = csv.writer(fout)
        writer.writerow(["id","created_at","text"])
        print ('Get tweet of user %s' % screen_name) 
        #make initial request for most recent tweets (200 is the maximum allowed count)
        try:
            new_tweets = api.user_timeline(screen_name = screen_name,count=200)
        except Exception as e:
            print ('Error with user ' + screen_name)
            print (e)
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
                print ('Error ' + screen_name)
                print (e)
                break
                
        fout.close()
    def download_tweets_dataset(self):
        if not os.path.isdir(self.tweet_dir):
            os.system('mkdir ' + self.tweet_dir)
        api = self.get_tweet_api()
        df_user_id = pandas.read_csv('userid_labels_screenname.csv',dtype=str)
        missing_user = []
        for user_id in df_user_id.sns_id:
            try:
                user_profile = api.lookup_users(user_ids=[user_id])
                file_path = os.path.join(self.tweet_dir, '%s_%s.csv' % (user_profile[0].screen_name, user_id))
                if not os.path.isfile(file_path):
                    self.get_user_tweets(user_profile[0].screen_name, file_path, api)
            except Exception as e:
                print ('Error with user ' + user_id)
                print (e)
                missing_user.append(user_id)
        
        print ('User  with missing data are:')
        print (missing_user)
        print ('Done downloading all tweets')    
    def preprocess_text(self, text):
        text = preprocessor.clean(text).lower()
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        text = regex.sub('', text)
        en_dict = enchant.Dict('en-US')
        text = ' '.join([w if en_dict.check(w) and not w.isdigit() else '' for w in text.split(' ')  if w != ''])
        return text
    
    def create_feature_matrix(self):
        #Load tweets and create the feature matrix for all users
        if not os.path.isdir(self.tweet_dir):
            os.system('mkdir ' + self.tweet_dir)
        api = self.get_tweet_api()
        user_labels = pandas.read_csv(os.path.join(self.home_dir, 'userid_labels_screenname.csv'), dtype=str)
        user_labels = user_labels.loc[user_labels.screen_name.notnull(), :]
        self.category_corpus = dict()
        def get_corpus(df_users):
            all_users_tweets = []
            for ind, row in df_users.iterrows():
                try:
                    file_path = os.path.join(self.tweet_dir, '%s_%s.csv' % (row.screen_name, row.sns_id))
                    if not os.path.isfile(file_path):
                        self.get_user_tweets(row.screen_name, file_path, api)
                    
                    df_tweet = pandas.read_csv(file_path, dtype=str)
                    if df_tweet.shape[0] > 0:
                        user_tweets = ' '.join(df_tweet.text[df_tweet.text.notnull()].tolist())
                        user_tweets = self.preprocess_text(user_tweets)
                        all_users_tweets.append(user_tweets)
                except Exception as e:
                    print ('Error ' + row.screen_name)
                    print (e)
            self.category_corpus[df_users.Category.iloc[0]] = all_users_tweets
        
        #Method 2: Category-based method 
        user_labels.groupby('Category').apply(get_corpus)
        categories = list(self.category_corpus.keys())
        tfidf = TfidfVectorizer(min_df=1, sublinear_tf=True, stop_words='english')
        categories_tweets = [' '.join(t) for t in self.category_corpus.values()]
        feature_mat_cat = tfidf.fit_transform(categories_tweets).toarray()
        pickle.dump(tfidf, open('tfidf_catbased.pickle', 'wb'))

        terms = numpy.array(tfidf.get_feature_names())
        #Find the terms that appear in only 1 single Category
        unique_terms_in_cat = numpy.sum(feature_mat_cat, axis=0) == numpy.max(feature_mat_cat, axis=0)
        #Only keep the scores of the unique features
        for i in range(len(categories)):
            feature_mat_cat[i] = feature_mat_cat[i] * unique_terms_in_cat

        print ('Top 10 terms in each category')
        for i in range(len(categories)):
            print (categories[i])
            best_feature_ind = numpy.argsort(-feature_mat_cat[i])[:10]
            print (terms[best_feature_ind])
         
        #Create feature matrix for all users
        labels = []
        user_tweets = []
        for key, val in self.category_corpus.items():
            labels += [key] * len(val)
            user_tweets += val
        feature_mat_user = tfidf.transform(user_tweets)
        pickle.dump(feature_mat_user, open('feature_mat_catbased.pickle', 'wb'))
        pickle.dump(labels, open('labels.pickle', 'wb'))
        feature_scores = numpy.max(feature_mat_cat, axis=0)
        pickle.dump(feature_scores, open('feature_scores.pickle', 'wb'))
        
        #Method 1: User-based method 
        tfidf = TfidfVectorizer(min_df=1, sublinear_tf=True, stop_words='english')
        feature_mat_user = tfidf.fit_transform(user_tweets)
        pickle.dump(feature_mat_user, open('feature_mat_userbased.pickle', 'wb'))
        pickle.dump(tfidf, open('tfidf_userbased.pickle', 'wb'))
        print ('Done creating feature matrices, which are saved in pickle files')

    def run(self, mode='test', user_id=None):
        if mode == 'feature':
            print ('Create feature matrix, find the best terms representing each category')
            self.create_feature_matrix()
            return
        if mode == 'download':
            print ('Download tweets of users stored in file userid_labels_screenname.csv')
            self.download_tweets_dataset()
            return 

        tfidf_filename = 'tfidf_catbased.pickle'
        feature_mat_filename = 'feature_mat_catbased.pickle'
        label_filename = 'labels.pickle'
        feature_score_filename = 'feature_scores.pickle'
        if not (os.path.isfile(tfidf_filename) and os.path.isfile(feature_mat_filename) and os.path.isfile(label_filename) and os.path.isfile(feature_score_filename)):
            print ('Features have not been generated yet, please run with mode "feature" first')
            return
        
        tfidf = pickle.load(open(os.path.join(self.home_dir, tfidf_filename), 'rb'))
        feature_mat = pickle.load(open(os.path.join(self.home_dir, feature_mat_filename), 'rb'))
        labels = pickle.load(open(os.path.join(self.home_dir, label_filename), 'rb'))
        feature_scores = pickle.load(open(os.path.join(self.home_dir, feature_score_filename), 'rb'))

        labels = [1 if i == 'Politician' else 2 if i == 'Trader' else 3 for i in labels]
        X_train, X_test, y_train, y_test = train_test_split(feature_mat, labels, test_size=0.3, random_state=0)
        if mode == 'model':
            print ('Model selection on the training set (70% of data)')
            #Model selection on the training set
            models = dict()
            models['NaiveBayes'] =  MultinomialNB()
            models['SVM'] = SVC(kernel='linear', C=1)
            models['DecisionTree'] = tree.DecisionTreeClassifier()
            models['RandomForest'] = RandomForestClassifier(n_estimators=100)
            #sort feature descending
            sorted_ind = numpy.argsort(-feature_scores)
            #Keep top k% of the features
            sel_feature_portion = [1, 0.9, 0.5, 0.3, 0.1]
            df_cv_results = pandas.DataFrame(index=[str(i) for i in sel_feature_portion], columns=models.keys())
            for portion in sel_feature_portion:
                remove_feature_ind = sorted_ind[int(portion*len(sorted_ind)):]
                for method_name, algo in models.items():
                    sel_features = numpy.delete(X_train.toarray(), remove_feature_ind, axis=1)
                    predicted = cross_val_predict(algo, sel_features, y_train, cv=5)
                    accuracy = metrics.accuracy_score(predicted, y_train)
                    df_cv_results.loc[str(portion), method_name] = accuracy
                    print ('Select top {:.0f}% portion of features, {:s}: {:.2f} accuracy'.format(portion*100, method_name, accuracy))
            
            df_cv_results.to_csv('cross_validation_training.csv')
            print ('Done. Model  selection results are:')
            print (df_cv_results.head)
        elif mode == 'test':
            print ('Test on 30% of data')
            SVM_model = SVC(kernel='linear', C=1)
            SVM_model.fit(X_train, y_train) 
            predicted = SVM_model.predict(X_test)
            accuracy = metrics.accuracy_score(predicted, y_test)
            f1 = metrics.f1_score(predicted, y_test, average='macro') #average of all classes
            print ('Accuracy, f1_score on the test set, including %d users is: %.2f, %.2f' % (len(y_test), accuracy, f1))
        else:
            print ('Predict  class probability of a user with id provided')
            if user_id is None:
                print ('User id not provided')
                return
            print ('Predict for user with id %s' % user_id)
            model_path = os.path.join(self.home_dir, 'model.pickle')
            if os.path.isfile(model_path):
                model = pickle.load(open(model_path, 'rb'))
            else:
                model = SVC(kernel='linear', C=1, probability=True)
                model.fit(feature_mat, labels)
                pickle.dump(model, open(model_path, 'wb'))
            api = self.get_tweet_api()
            try:
                user_profile = api.lookup_users(user_ids=[user_id])
            except Exception as e:
                print ('Cannot download user information')
                print (e)
                print ('Predict based on prior probability: 0.22 Politician, 0.31 Trader, and 0.47 Journalist')
                return
            
            file_path = os.path.join(self.tweet_dir, '%s_%s.csv' % (user_profile[0].screen_name, user_id))
            self.get_user_tweets(user_profile[0].screen_name, file_path, api)
            df_tweet = pandas.read_csv(file_path, dtype=str)
            if df_tweet.shape[0] == 0:
                print ('Cannot download tweets from this user')
                print ('Predict based on prior probability: 0.22 Politician, 0.31 Trader, and 0.47 Journalist')
                return
            else:
                user_tweets = ' '.join(df_tweet.text[df_tweet.text.notnull()].tolist())
                user_tweets = self.preprocess_text(user_tweets)
            
            user_feature = tfidf.transform([user_tweets])
            predicted = model.predict_proba(user_feature)
            print ('Prediction result of this user is: %.2f Politician, %.2f Trader, and %.2f Journalist' % (predicted[0][0], predicted[0][1], predicted[0][2]))
            return

if __name__ == '__main__':
    optparser = OptionParser()
    optparser.add_option('-p', dest='home_dir', default='.')
    optparser.add_option('-t', dest='tweet_dir', default='tweet_ascii', help='Folder to store downloaded user  tweets')
    optparser.add_option('-m', dest='mode', help='running mode: download (download tweets of users in the dataset), feature (create feature matrix), model (model selection on the training set), test (evaluate model on the test set - default option), predict (predict class probability of a new user)', default='test')
    optparser.add_option('-u', dest='user_id', help='id of the user for prediction (in case mode=predict)', default=None)
    (options, args) = optparser.parse_args()
    user = UserAnalysis(options.home_dir, options.tweet_dir)
    user.run(options.mode, options.user_id)

import matplotlib.pyplot as plt
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
        df_users = pandas.read_csv(os.path.join(self.home_dir, 'userid_labels_screenname.csv'), dtype=str)
        df_users = df_users.loc[df_users.screen_name.notnull(), :]
        #Load user tweets and labels
        user_tweets = []  
        labels = []
        for ind, row in df_users.iterrows():
            try:
                file_path = os.path.join(self.tweet_dir, '%s_%s.csv' % (row.screen_name, row.sns_id))
                if not os.path.isfile(file_path):
                    self.get_user_tweets(row.screen_name, file_path, api)
                
                df_tweet = pandas.read_csv(file_path, dtype=str)
                if df_tweet.shape[0] > 0:
                    tweets = ' '.join(df_tweet.text[df_tweet.text.notnull()].tolist())
                    tweets = self.preprocess_text(tweets)
                    user_tweets.append(tweets)
                    labels.append(row.Category)
            except Exception as e:
                print ('Error processing with user ' + row.screen_name)
                print (e)
        categories = numpy.unique(labels)
        ##########################################################################
        ##1. Split into training and test set
        tweet_train, tweet_test, y_train, y_test = train_test_split(numpy.array(user_tweets), numpy.array(labels), test_size=0.3, random_state=0)
        #######################################################################        
        #Method 1: User-based method 
        #Create the feature matrix for  all users using tfidf obtained from users
        tfidf = TfidfVectorizer(min_df=1, sublinear_tf=True, stop_words='english')
        #Learn dictionary from the training tweets
        X_train = tfidf.fit_transform(tweet_train)
        #Apply this dictionary to transform the test tweets
        X_test = tfidf.transform(tweet_test)
        
        pickle.dump(X_train, open('X_train_userbased.pickle', 'wb'))
        pickle.dump(X_test, open('X_test_userbased.pickle', 'wb'))
        pickle.dump(tfidf, open('tfidf_userbased.pickle', 'wb'))
        pickle.dump(y_train, open('y_train.pickle', 'wb'))
        pickle.dump(y_test, open('y_test.pickle', 'wb'))

        #Method 2: Category-based method 
        ###########################################################################
        tfidf = TfidfVectorizer(min_df=1, sublinear_tf=True, stop_words='english')
        tweet_cat = [' '.join(tweet_train[y_train == cat]) for cat in categories]
        
        #Learn tfidf from category-wise perspective, results are the tfidf of the term wrt to each category
        feature_mat_cat = tfidf.fit_transform(tweet_cat).toarray()
        feature_scores = numpy.max(feature_mat_cat, axis=0)
        #Apply the dictionary on the user tweets
        X_train = tfidf.transform(tweet_train)
        X_test = tfidf.transform(tweet_test)
        pickle.dump(X_train, open('X_train_catbased.pickle', 'wb'))
        pickle.dump(X_test, open('X_test_catbased.pickle', 'wb'))
        pickle.dump(tfidf, open('tfidf_catbased.pickle', 'wb'))
        pickle.dump(feature_scores, open('feature_scores.pickle', 'wb'))
  
        #************************************************************************************ 
        #Find the terms that appear in only 1 single Category
        terms = numpy.array(tfidf.get_feature_names())
        unique_terms_in_cat = numpy.sum(feature_mat_cat, axis=0) == numpy.max(feature_mat_cat, axis=0)
        #Only keep the scores of the unique features
        for i in range(len(categories)):
            feature_mat_cat[i] = feature_mat_cat[i] * unique_terms_in_cat

        print ('Top 10 terms in each category')
        for i in range(len(categories)):
            print (categories[i])
            best_feature_ind = numpy.argsort(-feature_mat_cat[i])[:10]
            print (terms[best_feature_ind])
         
    def test_SVM_param(self, X_train, y_train):
        print ('Evaluate different C values of SVM')
        C_params = [0.01, 0.1, 1, 10, 100] 
        accuracies = []
        f1_scores = []
        for C in C_params:
            SVM_model = SVC(kernel='linear', C=C)
            predicted = cross_val_predict(SVM_model, X_train, y_train, cv=3)
            accuracies.append(metrics.accuracy_score(predicted, y_train))
            f1_scores.append(metrics.f1_score(predicted, y_train, average='macro')) #average of all classes

        plt.plot(range(len(C_params)), accuracies, 'r', range(len(C_params)), f1_scores, 'b', linewidth=2.0)
        plt.xticks(range(len(C_params)), [str(i) for i in (C_params)])
        plt.legend(['Accuracy', 'F1 measure'], loc='upper_left')
        plt.xlabel('Parameter C')
        plt.ylabel('Result')
        plt.show()
    def run(self, mode='test', user_id=None, idf_catbased=True):
        if mode == 'feature':
            print ('Create feature matrix, find the best terms representing each category')
            self.create_feature_matrix()
            return
        if mode == 'download':
            print ('Download tweets of users stored in file userid_labels_screenname.csv')
            self.download_tweets_dataset()
            return 
        cv_result_file = 'cross_validation_' 
        try: 
            #Load files 
            y_train = pickle.load(open('y_train.pickle', 'rb'))
            y_test = pickle.load(open('y_test.pickle', 'rb'))
            y_train = [1 if i == 'Politician' else 2 if i == 'Trader' else 3 for i in y_train]
            y_test = [1 if i == 'Politician' else 2 if i == 'Trader' else 3 for i in y_test]

            feature_scores = pickle.load(open('feature_scores.pickle', 'rb'))
            if idf_catbased:
                X_train = pickle.load(open('X_train_catbased.pickle', 'rb'))
                X_test = pickle.load(open('X_test_catbased.pickle', 'rb'))
                tfidf = pickle.load(open('tfidf_catbased.pickle', 'rb'))
                cv_result_file += '_catbased.csv'
            else:
                X_train = pickle.load(open('X_train_userbased.pickle', 'rb'))
                X_test = pickle.load(open('X_test_userbased.pickle', 'rb'))
                tfidf = pickle.load(open('tfidf_userbased.pickle', 'rb'))
                cv_result_file += '_userbased.csv'

        except Exception as e:
            print ('Cannot load feature files. Please run feature extraction first')
            return
        #sort feature descending
        sorted_ind = numpy.argsort(-feature_scores)
        sel_feature_portion = [1, 0.9, 0.5, 0.3, 0.1]
        if mode == 'model':
 
            print ('Model selection on the training set (70% of data)')
            #Model selection on the training set
            models = dict()
            models['NaiveBayes'] =  MultinomialNB()
            models['SVM'] = SVC(kernel='linear', C=1)
            models['DecisionTree'] = tree.DecisionTreeClassifier()
            models['RandomForest'] = RandomForestClassifier(n_estimators=100)
            #Keep top k% of the features
 
            df_cv_results = pandas.DataFrame(index=[str(i) for i in sel_feature_portion], columns=models.keys())
            for portion in sel_feature_portion:
                remove_feature_ind = sorted_ind[int(portion*len(sorted_ind)):]
                for method_name, algo in models.items():
                    sel_features = numpy.delete(X_train.toarray(), remove_feature_ind, axis=1)
                    predicted = cross_val_predict(algo, sel_features, y_train, cv=5)
                    accuracy = metrics.accuracy_score(predicted, y_train)
                    df_cv_results.loc[str(portion), method_name] = accuracy
                    print ('Select top {:.0f}% portion of features, {:s}: {:.2f} accuracy'.format(portion*100, method_name, accuracy))
            
            df_cv_results.to_csv(cv_result_file)
            print ('Done. Model  selection results are:')
            print (df_cv_results.head)
            self.test_SVM_param(X_train, y_train)
            return
        if  mode == 'test' or mode == 'predict':
            portion = 0.9
            remove_feature_ind = sorted_ind[int(portion*len(sorted_ind)):]
            model = SVC(kernel='linear', C=1, probability=True)

            if mode == 'test':
                print ('Test on 30% of data')
                sel_features_train = numpy.delete(X_train.toarray(), remove_feature_ind, axis=1)
                model.fit(sel_features_train, y_train) 
                sel_features_test = numpy.delete(X_test.toarray(), remove_feature_ind, axis=1)
                
                predicted = model.predict(sel_features_test)
                accuracy = metrics.accuracy_score(predicted, y_test)
                f1_score  = metrics.f1_score(predicted, y_test, average='macro') #average of all classes
                print ('Use %.2f portion of features. Accuracy, f1_score on the test set, including %d users is: %.2f, %.2f' % (portion, len(y_test), accuracy, f1_score))

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
                    #Train the final model using  all users data
                    X_train = numpy.concatenate((X_train.toarray(), X_test.toarray()), axis=0)
                    y_train += y_test
                    sel_features_train = numpy.delete(X_train, remove_feature_ind, axis=1)
                    model.fit(sel_features_train, y_train) 
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
                user_feature = numpy.delete(user_feature.toarray(), remove_feature_ind, axis=1)
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
    idf_catbased = True
    user.run(options.mode, options.user_id, idf_catbased)

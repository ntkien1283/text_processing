# tweet_user_classification
A python module to categorize tweeter users into into 3 classes: (A) Political oriented, (B) Financial oriented (C) Typical/Normal
The code can be run as follow (use python3) python3 user_analysis.py -m [mode] -u [user_id] - Possible values for  mode  are:
+ download  (to download tweet data from user dataset)
+ feature  (to extract features by both method 1 and 2 above and save to pickle files)
+ model  (for model and feature selection, using method 2 by default)
+ test  (train on the training set and evaluate on the test set)
+ predict  (to predict the probability of each class for a user)
- user_id: the id of the user to be predicted

Examples:

To download tweet datas:
Python3 user_analysis.py -m download

To extract features:
Python3 user_analysis.py -m feature

To perform model selection:
Python3 user_analysis.py -m model

To evaluate the model on the test dataset: 
Python3 user_analysis.py -m test

To predict the class of a user with id:
Python3 user_analysis.py -m predict -u 4872354465

See summary.pdf for more details

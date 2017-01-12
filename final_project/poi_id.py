#!/usr/bin/python

import sys
import pickle
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".

features_list = ['poi',
                 'bonus',
                 'deferred_income',
                 'exercised_stock_options',
                 'relative_messages_to_poi',
                 'salary',
                 'total_stock_value']

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Task 2: Remove outliers / data preparation
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']  # See notebook data exploration

for outlier in outliers:
    del data_dict[outlier]

# Task 3: Create new feature(s)
for employee in data_dict:
    to_messages = data_dict[employee]['to_messages']
    from_messages = data_dict[employee]['from_messages']
    from_poi_to_this_person = data_dict[employee]['from_poi_to_this_person']
    from_this_person_to_poi = data_dict[employee]['from_this_person_to_poi']

    # If either of the input features for the new feature is 'NaN',
    # the result for the new feature will also be 'NaN'
    if to_messages == 'NaN' or from_poi_to_this_person == 'NaN':
        data_dict[employee]['relative_messages_to_poi'] = 'NaN'
    if from_messages == 'NaN' or from_this_person_to_poi == 'NaN':
        data_dict[employee]['relative_messages_from_poi'] = 'NaN'

    # If both input features are not 'NaN', calculate the new features
    else:
        # Convert to float here, otherwise the check for 'NaN' above would not have worked correctly
        # (values would be the 'real' nan)
        to_messages = float(to_messages)
        from_messages = float(from_messages)
        from_poi_to_this_person = float(from_poi_to_this_person)
        from_this_person_to_poi = float(from_this_person_to_poi)

        # Calculate the new features
        data_dict[employee]['relative_messages_from_poi'] = from_poi_to_this_person/to_messages
        data_dict[employee]['relative_messages_to_poi'] = from_this_person_to_poi/from_messages

# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# from sklearn.feature_selection import f_classif, SelectKBest
# selector = SelectKBest(f_classif, k=10)
# selected_features_train = selector.fit_transform(features_train, labels_train)

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

# DECISION TREE
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.grid_search import GridSearchCV
#
# tree = DecisionTreeClassifier(random_state=42)
# params = {'min_samples_split': [2, 4, 8, 16, 32],
#           'max_depth': [2, 3, 4, 5, 6]}
# clf = GridSearchCV(tree, param_grid=params)
# clf.fit(features_train, labels_train)
# print clf.best_params_
# print clf.scorer_
# clf = clf.best_estimator_

# # SVM
# from sklearn.svm import SVC
# from sklearn.grid_search import GridSearchCV
# from sklearn.preprocessing import MinMaxScaler
# svm = SVC(random_state=42)
# scaler = MinMaxScaler()
#
# features_train = scaler.fit_transform(features_train)
#
# params = {'C': [0.01, 0.1, 1, 1, 10, 100, 100],
#           'gamma': [0.01, 0.1, 1, 1, 10, 100, 100]}
#
# clf = GridSearchCV(svm, param_grid=params)
# clf.fit(features_train, labels_train)
# print clf.best_params_
# clf = clf.best_estimator_

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

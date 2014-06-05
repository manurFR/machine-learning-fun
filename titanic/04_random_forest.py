#!/usr/bin/env python
# -*- coding: UTF8 -*-

import csv, sys
import numpy as np
import scipy as sp
import pandas as pan
from sklearn.cross_validation import KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from matplotlib import pylab
from utils import load_train_data, add_sex_bit, fill_fare, output_predictions, fill_median_age

best_score = 0.0
best_classifier = None
best_classifier_name = None

def test_algo(algo, features, classes, name, options={}):
	cv = KFold(n=len(features), n_folds=8, indices=True)

	scores = []
	for train, test in cv:
		X_train, y_train = features.values[train], classes.values[train]
		X_test, y_test = features.values[test], classes.values[test]

		classifier = algo(**options)
		classifier.fit(X_train, y_train)

		scores.append(classifier.score(X_test, y_test))

	score = np.mean(scores)
	print "Score on training set (with cross-validation) for %s : %.4f" % (name, score)
	# with open('decisiontree.dot', 'w') as f:
	# 	f = export_graphviz(classifier, out_file = f, feature_names = X.columns.tolist())
	global best_score, best_classifier, best_classifier_name
	if score >= best_score:
		best_score = score
		best_classifier = classifier
		best_classifier_name = name

df, X, Y = load_train_data()
X = X[['Pclass', 'SibSp', 'Parch', 'Fare', 'SexBit', 'AgeFill']]

test_algo(DecisionTreeClassifier, X, Y, "Decision Tree")
test_algo(DecisionTreeClassifier, X, Y, "Decision Tree with entropy criterion", {'criterion': 'entropy'})
test_algo(DecisionTreeClassifier, X, Y, "Decision Tree with all features", {'max_features': None})
test_algo(DecisionTreeClassifier, X, Y, "Decision Tree with max_depth=5", {'max_depth': 5})
test_algo(DecisionTreeClassifier, X, Y, "Decision Tree with min_samples_leaf=5", {'min_samples_leaf': 5})

print
for min_samples in range(1,20):
	test_algo(DecisionTreeClassifier, X, Y, "Decision Tree with max_depth=10 and min_samples_leaf=%d" % min_samples, {'max_depth': 10, 
																													  'min_samples_leaf': min_samples})
print
# min_samples_leaf = 8 is the best
for max_depth in range(1,20):
	test_algo(DecisionTreeClassifier, X, Y, "Decision Tree with max_depth=%d and min_samples_leaf=8" % max_depth, {'max_depth': max_depth, 
																												   'min_samples_leaf': 8})
# max_depth = 10 seems to stay the best
print

test_algo(RandomForestClassifier, X, Y, "Random Forest")
test_algo(RandomForestClassifier, X, Y, "Random Forest whose trees have max_depth=10 and min_samples_leaf=8", 
			{'max_depth': 10, 'min_samples_leaf': 8})
test_algo(RandomForestClassifier, X, Y, "Random Forest with 50 trees", {'n_estimators': 50})
test_algo(RandomForestClassifier, X, Y, "Random Forest with 100 trees", {'n_estimators': 100})
test_algo(RandomForestClassifier, X, Y, "Random Forest with 100 trees and the best params for trees", 
			{'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 8})
print

def extract_title(s):
	if 'Mr.' in s: 
		return 'Mr.'
	if 'Mrs.' in s:
		return 'Mrs.'
	if 'Miss.' in s:
		return 'Miss'
	if 'Rev.' in s:
		return 'Rev.'
	if 'Major.' in s:
		return 'Major.'
	if 'Lady.' in s:
		return 'Lady.'
	return 'Other'

embarked_encoder = LabelEncoder()
X['Embarked'] = embarked_encoder.fit_transform(df['Embarked'])
ticket_encoder = LabelEncoder()
X['Ticket'] = ticket_encoder.fit_transform(df['Ticket'].str[0])
title_encoder = LabelEncoder()
X['Title'] = title_encoder.fit_transform(df['Name'].apply(lambda name: extract_title(name)))

test_algo(DecisionTreeClassifier, X, Y, "Decision Tree with best params and Embarked, Ticket and Title", {'max_depth': 10, 'min_samples_leaf': 8})
test_algo(RandomForestClassifier, X, Y, "Random Forest with best params, 100 trees, and Embarked, Ticket and Title", 
			{'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 8})

print
print '='*100
print

print "Best overall: %s with %.4f" % (best_classifier_name, best_score)

output_predictions(best_classifier, '04_submission.csv', ['Pclass', 'SexBit', 'AgeFill', 'SibSp', 'Parch', 'Fare'],
				   [add_sex_bit, fill_fare, fill_median_age])
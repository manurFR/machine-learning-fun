#!/usr/bin/env python
# -*- coding: UTF8 -*-

import csv, sys
import numpy as np
import scipy as sp
import pandas as pan
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from matplotlib import pylab
from utils import load_train_data, add_sex_bit, fill_fare, fill_median_age, output_predictions, plot_learning_curve

formatting_functions = [add_sex_bit, fill_fare, fill_median_age]
features = ['Pclass', 'SibSp', 'Parch', 'Fare', 'SexBit', 'AgeFill']

best_score = 0.0
best_classifier = ()

def test_algo(algo, data, classes, name, options={}):
	cv = KFold(n=len(data), n_folds=8, indices=True)

	scores = []
	for train, test in cv:
		X_train, y_train = data.values[train], classes.values[train]
		X_test, y_test = data.values[test], classes.values[test]

		classifier = algo(random_state = 0, **options)
		classifier.fit(X_train, y_train)

		scores.append(classifier.score(X_test, y_test))

	score = np.mean(scores)
	print "Score on training set (with cross-validation) for %s : %.5f" % (name, score)
	# with open('decisiontree.dot', 'w') as f:
	# 	f = export_graphviz(classifier, out_file = f, feature_names = X.columns.tolist())
	global best_score, best_classifier, best_classifier_name
	if score >= best_score:
		best_score = score
		best_classifier = (classifier, algo, options, name)

df, X, Y = load_train_data(formatting_functions)
X = X[features]

test_algo(DecisionTreeClassifier, X, Y, "Decision Tree")
test_algo(DecisionTreeClassifier, X, Y, "Decision Tree with entropy criterion", {'criterion': 'entropy'})
test_algo(DecisionTreeClassifier, X, Y, "Decision Tree with sqrt(features)", {'max_features': 'sqrt'})
test_algo(DecisionTreeClassifier, X, Y, "Decision Tree with max_depth=5", {'max_depth': 5})
test_algo(DecisionTreeClassifier, X, Y, "Decision Tree with min_samples_leaf=5", {'min_samples_leaf': 5})

print
for min_samples in range(1,21):
	for max_depth in range(1,21):
		test_algo(DecisionTreeClassifier, X, Y, "Decision Tree with max_depth=%d and min_samples_leaf=%d" % 
			(max_depth, min_samples), {'max_depth': max_depth, 'min_samples_leaf': min_samples})
print


classifier, algo, options, name = best_classifier
print "Best overall: %s with %.5f" % (name, best_score)

output_predictions(classifier, '04_submission.csv', formatting_functions, features)
plot_learning_curve(name, algo, options, X, Y, min_size=50, n_steps=50)

print '='*100
print

# Random forests

best_score = 0.0
best_classifier = ()

test_algo(RandomForestClassifier, X, Y, "Random Forest with 10 trees")
test_algo(RandomForestClassifier, X, Y, "Random Forest with 50 trees", {'n_estimators': 50})
test_algo(RandomForestClassifier, X, Y, "Random Forest with 100 trees", {'n_estimators': 100})
test_algo(RandomForestClassifier, X, Y, "Random Forest with 10 trees, max_depth=6 and min_samples_leaf=6", 
			{'max_depth': 6, 'min_samples_leaf': 6})
test_algo(RandomForestClassifier, X, Y, "Random Forest with 50 trees, max_depth=6 and min_samples_leaf=6", 
			{'n_estimators': 50, 'max_depth': 6, 'min_samples_leaf': 6})
test_algo(RandomForestClassifier, X, Y, "Random Forest with 100 trees, max_depth=6 and min_samples_leaf=6", 
			{'n_estimators': 100, 'max_depth': 6, 'min_samples_leaf': 6})
test_algo(RandomForestClassifier, X, Y, "Random Forest with 500 trees, max_depth=6 and min_samples_leaf=6", 
			{'n_estimators': 500, 'max_depth': 6, 'min_samples_leaf': 6})
print

classifier, algo, options, name = best_classifier
print "Best overall Random Forest: %s with %.5f" % (name, best_score)

output_predictions(classifier, '04_submission_rf.csv', formatting_functions, features)

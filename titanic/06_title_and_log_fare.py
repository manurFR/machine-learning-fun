#!/usr/bin/env python
# -*- coding: UTF8 -*-

import numpy as np
from utils import load_train_data, output_predictions, add_sex_bit, fill_fare, add_title, test_algo
from sklearn.tree import DecisionTreeClassifier

def logarize_fare(X):
	X['FareLog'] = np.log10(X['Fare'] + 1)

df, X, Y = load_train_data([add_sex_bit, fill_fare, add_title, logarize_fare])
add_title(X)
logarize_fare(X)

features = ['Pclass', 'SibSp', 'Parch', 'FareLog', 'SexBit', 'Title']

best_run = [0.0, 0, 0] # [score, max_depth, min_samples]
for min_samples in range(1,21):
	for max_depth in range(1,21):
		score = test_algo(DecisionTreeClassifier, X[features], Y, "Decision Tree with max_depth=%d and min_samples_leaf=%d" % 
								(max_depth, min_samples), {'max_depth': max_depth, 'min_samples_leaf': min_samples})
		if score > best_run[0]:
			best_run = [score, max_depth, min_samples]

print
score, max_depth, min_samples = best_run
print "Best overall: max_depth=%d, min_samples=%d with score=%.5f" % (max_depth, min_samples, score)

# test_algo(DecisionTreeClassifier, X[features], Y, "Decision Tree with Title instead of Age", {'max_depth': 10, 'min_samples_leaf': 8})

classifier = DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_samples)
classifier.fit(X[features], Y)

output_predictions(classifier, "06_submission.csv", [add_sex_bit, fill_fare, add_title, logarize_fare], features)

print
print "Importance of features:"
print features
print classifier.feature_importances_
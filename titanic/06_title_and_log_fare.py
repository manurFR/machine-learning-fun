#!/usr/bin/env python
# -*- coding: UTF8 -*-

import numpy as np
from utils import load_train_data, output_predictions, add_sex_bit, fill_fare, add_title, test_algo
from sklearn.tree import DecisionTreeClassifier

def logarize_fare(X):
	X['FareLog'] = np.log10(X['Fare'] + 1)

df, X, Y = load_train_data()
add_title(X)
logarize_fare(X)

features = ['Pclass', 'SibSp', 'Parch', 'FareLog', 'SexBit', 'Title']

test_algo(DecisionTreeClassifier, X[features], Y, "Decision Tree with Title instead of Age", {'max_depth': 10, 'min_samples_leaf': 8})

classifier = DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 8)
classifier.fit(X[features], Y)

output_predictions(classifier, "06_submission.csv", features, [add_sex_bit, fill_fare, add_title, logarize_fare])

print "Importance of features:"
print features
print classifier.feature_importances_
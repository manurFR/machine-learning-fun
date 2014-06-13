#!/usr/bin/env python
# -*- coding: UTF8 -*-

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from utils import load_train_data, add_sex_bit, fill_fare, fill_median_age, output_predictions, plot_learning_curve, \
				  encode_embarked, extract_deck, add_title, logarize_fare, test_algo


# add title, embarked, deck
formatting_functions1 = [add_sex_bit, fill_fare, fill_median_age, encode_embarked, extract_deck, add_title]
features1 = ['Pclass', 'SibSp', 'Parch', 'Fare', 'SexBit', 'AgeFill', 'Embarked', 'Deck', 'Title']

df, X, Y = load_train_data(formatting_functions1)
X = X[features1]

# test_algo(RandomForestClassifier, X, Y, "Random Forest with Embarked, Deck and Title", 
# 			{'n_estimators': 200, 'max_depth': 6, 'min_samples_leaf': 6, 'random_state': 2})

test_algo(DecisionTreeClassifier, X, Y, "Decision Tree with Embarked, Deck and Title", 
			{'max_depth': 6, 'min_samples_leaf': 6, 'random_state': 2})


# classifier = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=6)
classifier = DecisionTreeClassifier(max_depth=6, min_samples_leaf=6)
classifier.fit(X, Y)

print "Importance of features:"
print features1
print classifier.feature_importances_

output_predictions(classifier, '05_submission_1.csv', formatting_functions1, features1)
print

# title instead of age (no embarked nor deck)
formatting_functions2 = [add_sex_bit, fill_fare, add_title]
features2 = ['Pclass', 'SibSp', 'Parch', 'Fare', 'SexBit', 'Title']

df, X, Y = load_train_data(formatting_functions2)
X = X[features2]

# test_algo(RandomForestClassifier, X, Y, "Random Forest with Title and no Age", 
# 			{'n_estimators': 200, 'max_depth': 6, 'min_samples_leaf': 6, 'random_state': 2})
test_algo(DecisionTreeClassifier, X, Y, "Decision Tree with Title and no Age", 
			{'max_depth': 6, 'min_samples_leaf': 6, 'random_state': 2})

# classifier = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=6)
classifier = DecisionTreeClassifier(max_depth=6, min_samples_leaf=6)
classifier.fit(X, Y)

print "Importance of features:"
print features2
print classifier.feature_importances_

output_predictions(classifier, '05_submission_2.csv', formatting_functions2, features2)
print

# title instead of age (no embarked nor deck) but log(fare+1) instead of fare
formatting_functions3 = [add_sex_bit, fill_fare, logarize_fare, add_title]
features3 = ['Pclass', 'SibSp', 'Parch', 'LogFarePlusOne', 'SexBit', 'Title']

df, X, Y = load_train_data(formatting_functions3)
X = X[features3]

# test_algo(RandomForestClassifier, X, Y, "Random Forest with Title, Log(Fare+1) and no Age", 
# 			{'n_estimators': 200, 'max_depth': 6, 'min_samples_leaf': 6, 'random_state': 2})
test_algo(DecisionTreeClassifier, X, Y, "Decision Tree with Title, Log(Fare+1) and no Age", 
			{'max_depth': 6, 'min_samples_leaf': 6, 'random_state': 2})

# classifier = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=6)
classifier = DecisionTreeClassifier(max_depth=6, min_samples_leaf=6)
classifier.fit(X, Y)

print "Importance of features:"
print features3
print classifier.feature_importances_

output_predictions(classifier, '05_submission_3.csv', formatting_functions3, features3)

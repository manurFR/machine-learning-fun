#!/usr/bin/env python
# -*- coding: UTF8 -*-

import numpy as np
import pandas as pan
import re
from utils import load_train_data, plot_bias_variance, output_predictions
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold, train_test_split

df, X, Y, median_ages = load_train_data()

datasizes = np.arange(50, len(X), 30)

train_errors = []
cv_errors = []
for datasize in datasizes:
	X_train, X_cv, y_train, y_cv = train_test_split(X[:datasize], Y[:datasize], test_size=0.7)
	classifier = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=8)
	classifier.fit(X_train,y_train)
	train_errors.append(1 - classifier.score(X_train, y_train))
	cv_errors.append(1 - classifier.score(X_cv, y_cv))

#plot_bias_variance(datasizes, train_errors, cv_errors, "Learning curve (Bias/Variance detection) for Random Forest")	
# A small gap between train errors and cv errors stays even for large data size : maybe a bit of overfitting

def add_title(X):
	pattern = re.compile(r'.*(Mr|Mrs|Miss|Rev|Major|Col|Lady|Master|Dr).*')
	def extract_title(name):
		match = pattern.match(name)
		if match:
			return match.group(1)
		else:
			return 'Other'
	title_encoder = LabelEncoder()
	X['Title'] = title_encoder.fit_transform(df['Name'].apply(extract_title))
	return X
	

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

X = add_title(X)
X = X.drop(['AgeFill'], axis=1)

test_algo(DecisionTreeClassifier, X, Y, "Decision Tree with Title instead of Age", {'max_depth': 10, 'min_samples_leaf': 8})
#test_algo(RandomForestClassifier, X, Y, "Decision Tree with Title instead of Age", {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 8})

classifier = DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 8)
classifier.fit(X,Y)

def add_sex_bit(X):
	X['SexBit'] = X['Sex'].map({'female': 0, 'male':1}).astype(int)
	return X

def fill_fare(X):
	X['Fare'] = X['Fare'].fillna(0.0)
	return X

# Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
output_predictions(classifier, "05_submission.csv", [
	add_sex_bit,
	fill_fare,
	add_title,
	lambda Xtest: Xtest.drop(['Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked'], axis = 1)
])
# TODO ne plus appeler df dans add_title, et donc laisser 'Name' dans X dans load_train_data()
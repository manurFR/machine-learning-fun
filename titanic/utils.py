#!/usr/bin/env python
# -*- coding: UTF8 -*-

import pandas as pan
import numpy as np
import re
from matplotlib import pylab
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder

def add_sex_bit(X):
	X['SexBit'] = X['Sex'].map({'female': 0, 'male':1}).astype(int)

def fill_fare(X):
	X['Fare'] = X['Fare'].fillna(0.0)

def fill_median_age(X):
	median_ages = np.zeros((2,3))
	for i in range(2):
		for j in range(3):
			median_ages[i,j] = X[(X['SexBit'] == i) & (X['Pclass']-1 == j)]['Age'].dropna().median()

	X['AgeFill'] = X['Age']
	for i in range(2):
		for j in range(3):
			X.loc[(X.Age.isnull()) & (X.SexBit == i) & (X.Pclass-1 == j), 'AgeFill'] = median_ages[i,j]

def add_title(X, names=None):
	pattern = re.compile(r'.*(Mr|Mrs|Miss|Rev|Major|Col|Lady|Master|Dr).*')
	def extract_title(name):
		match = pattern.match(name)
		if match:
			return match.group(1)
		else:
			return 'Other'
	if names is None:
		names = X['Name']
	title_encoder = LabelEncoder()
	X['Title'] = title_encoder.fit_transform(names.apply(extract_title))

def load_train_data():
	#PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
	df = pan.read_csv('train.csv', header=0)

	Y = df['Survived']

	X = df[['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
	add_sex_bit(X)
	fill_fare(X)
	fill_median_age(X)

	return df, X, Y

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

def plot_bias_variance(data_sizes, train_errors, test_errors, title):
    pylab.figure(num=None, figsize=(6, 5))
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('Data set size')
    pylab.ylabel('Error')
    pylab.title(title)
    pylab.plot(
        data_sizes, test_errors, "--", data_sizes, train_errors, "b-", lw=1)
    pylab.legend(["train error", "test error"], loc="upper right")
    pylab.grid(True, linestyle='-', color='0.75')
    pylab.show()

def output_predictions(classifier, output_name, features=[], format_funcs = []):
	Xtest = pan.read_csv('test.csv', header=0)
	
	ids = Xtest['PassengerId'].values
	Xtest = Xtest.drop(['PassengerId'], axis=1)

	for func in format_funcs:
		func(Xtest)

	if features:
		Xtest = Xtest[features]
	predictions = classifier.predict(Xtest.values)

	with open(output_name, 'w') as f:
		f.write("PassengerId,Survived\n")
		for idx, pred in enumerate(predictions):
			f.write(str(ids[idx]) + "," + str(int(pred)) + "\n")
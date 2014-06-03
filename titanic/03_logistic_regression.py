#!/usr/bin/env python
# -*- coding: UTF8 -*-

import csv, sys
import numpy as np
import scipy as sp
import pandas as pan
import sklearn.linear_model, sklearn.preprocessing
from sklearn.cross_validation import KFold

def prepareFeatures(dataframe, median_ages=None):
	X = dataframe[['Pclass', 'Fare', 'Age']]

	X['SexBit'] = df['Sex'].map({'female': 0, 'male':1}).astype(int)
	X['PclassIdx'] = X['Pclass'].map(lambda x: int(x)-1).astype(int)
	X['EmbarkedCode'] = df['Embarked'].map({None: 1, 'C': 0, 'S': 1, 'Q': 2}).astype(int)
	X.loc[X['Fare'].isnull(), 'Fare'] = 0.0

	# Replace missing ages by median value (of age) for the sex and the pclass
	if median_ages is None:
		median_ages = np.zeros((2,3))
		for i in range(2):
			for j in range(3):
				median_ages[i,j] = X[(X['SexBit'] == i) & (X['PclassIdx'] == j)]['Age'].dropna().median()

	X['AgeFill'] = X['Age']
	for i in range(2):
		for j in range(3):
			X.loc[(X.Age.isnull()) & (X.SexBit == i) & (X.PclassIdx == j), 'AgeFill'] = median_ages[i,j]
	# X['AgeIsNull'] = pan.isnull(X.Age).astype(int)

	# Engineered features
	X['FamilySize'] = df['SibSp'] + df['Parch']
	X['Age*Class'] = X.AgeFill * df.Pclass

	# Drop useless string columns
	X = X.drop(['PclassIdx','Age'], axis=1)
	return X, median_ages

df = pan.read_csv('train.csv', header=0)

# survival status
Y = df['Survived']

X_pan, median_ages = prepareFeatures(df)

# For scikit-learn, we should convert the pandas dataframe to a numpy ndarray
X = X_pan.values
Y = Y.values

# Logistic Regression
best_score = 0.0
best_C = -1.0

for C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
	cv = KFold(n=len(X), n_folds=8, indices=True)

	scores = []
	for train, test in cv:
		X_train, y_train = X[train], Y[train]
		X_test, y_test = X[test], Y[test]

		lr = sklearn.linear_model.LogisticRegression(C=C)
		lr.fit(X_train,y_train)

		scores.append(lr.score(X_test, y_test))

	print "Score on training set (with cross-validation) for C=%f : %.4f" % (C, np.mean(scores))

	if np.mean(scores) > best_score:
		best_score = np.mean(scores)
		best_C = C

print "Best C =", best_C

lr = sklearn.linear_model.LogisticRegression(C = best_C)
lr.fit(X,Y)

print "intercept:", lr.intercept_
print "coefs:", lr.coef_
print "features:", X_pan.columns.tolist()

dfTest = pan.read_csv('test.csv', header=0)
test, _ = prepareFeatures(dfTest, median_ages)
Xtest = test.values

with open('03_submission.csv', 'w') as f:
	f.write("PassengerId,Survived\n")
	for idx, passenger in enumerate(Xtest):
		f.write(str(dfTest.loc[idx,'PassengerId']) + "," + str(int(lr.predict(passenger))) + "\n")



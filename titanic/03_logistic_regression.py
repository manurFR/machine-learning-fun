#!/usr/bin/env python
# -*- coding: UTF8 -*-

import csv, sys
import numpy as np
import scipy as sp
import pandas as pan
import sklearn.linear_model, sklearn.preprocessing
from sklearn.cross_validation import KFold, train_test_split
from matplotlib import pylab

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

def plot_best_regularization(C_values, scores):
    pylab.figure(num=None, figsize=(6, 5))
    pylab.ylim([0.79, 0.81])
    pylab.xlabel('C (regularization parameter)')
    pylab.ylabel('Score')
    pylab.title('Logistic Regression score for C')
    pylab.plot(C_values, scores, "-", lw=1)
    pylab.grid(True, linestyle='--', color='0.75')
    pylab.savefig('charts/logreg_bestC.png')

# Logistic Regression
C_values = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
C_scores = []
for C in C_values:
	cv = KFold(n=len(X), n_folds=8, indices=True)

	scores = []
	for train, test in cv:
		X_train, y_train = X[train], Y[train]
		X_test, y_test = X[test], Y[test]

		lr = sklearn.linear_model.LogisticRegression(C=C)
		lr.fit(X_train,y_train)

		scores.append(lr.score(X_test, y_test))

	C_scores.append(np.mean(scores))
	print "Score on training set (with cross-validation) for C=%f : %.5f" % (C, np.mean(scores))

best_C = C_values[np.argmax(C_scores)]
print "Best C =", best_C

plot_best_regularization(C_values, C_scores)

datasizes = np.arange(50, len(X), 20)

train_errors = []
cv_errors = []
for datasize in datasizes:
	X_train, X_cv, y_train, y_cv = train_test_split(X[:datasize], Y[:datasize], test_size=0.7)
	lr = sklearn.linear_model.LogisticRegression(C=best_C)
	lr.fit(X_train,y_train)
	train_errors.append(1 - lr.score(X_train, y_train))
	cv_errors.append(1 - lr.score(X_cv, y_cv))

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
    pylab.savefig('charts/logreg_bias_variance.png')

plot_bias_variance(datasizes, train_errors, cv_errors, "Learning curve for Titanic Logistic Regression")
# the curve shows we have high bias (we don't really fit the training set much better than the cross-validation set)
#  ==> we need a more complex model (non-linear?)
lr = sklearn.linear_model.LogisticRegression(C = best_C)
lr.fit(X,Y)

print "intercept:", lr.intercept_
print "coefs:", lr.coef_
print "features:", X_pan.columns.tolist()

dfTest = pan.read_csv('test.csv', header=0)
test, _ = prepareFeatures(dfTest, median_ages)
Xtest = test.values

predictions = lr.predict(Xtest)

with open('03_submission.csv', 'w') as f:
	f.write("PassengerId,Survived\n")
	for idx, pred in enumerate(predictions):
		f.write(str(dfTest.loc[idx,'PassengerId']) + "," + str(int(pred)) + "\n")



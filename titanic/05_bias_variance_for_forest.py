#!/usr/bin/env python
# -*- coding: UTF8 -*-

import numpy as np
from utils import load_train_data, plot_bias_variance, output_predictions, add_sex_bit, fill_fare, add_title, test_algo
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

df, X, Y = load_train_data()

names = X['Name']
X = X[['Pclass', 'SibSp', 'Parch', 'Fare', 'SexBit', 'AgeFill']]

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

add_title(X, names)
X = X.drop(['AgeFill'], axis=1)

test_algo(DecisionTreeClassifier, X, Y, "Decision Tree with Title instead of Age", {'max_depth': 10, 'min_samples_leaf': 8})
#test_algo(RandomForestClassifier, X, Y, "Decision Tree with Title instead of Age", {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 8})

classifier = DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 8)
classifier.fit(X, Y)

# Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
features = ['Pclass', 'SibSp', 'Parch', 'Fare', 'SexBit', 'Title']
output_predictions(classifier, "05_submission.csv", features, [add_sex_bit, fill_fare, add_title])

print "Importance of features:"
print features
print classifier.feature_importances_
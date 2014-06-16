#!/usr/bin/env python
# -*- coding: UTF8 -*-

import numpy as np
import scipy as sp
from utils import load_train_data, test_algo, output_predictions, add_sex_bit, fill_fare, add_title, logarize_fare
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

formatting = [add_sex_bit, fill_fare, add_title, logarize_fare]
features = ['Pclass', 'SibSp', 'Parch', 'SexBit', 'Title', 'LogFarePlusOne']

df, X, Y = load_train_data(formatting)

# SVM
best_run = [0.0, 0.0] # score, C
for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 5.0, 10.0, 15.0, 30.0, 100.0]:
	score = test_algo(SVC, X[features], Y, "SVM with C=%5g" % C, {'C': C, 'random_state': 0})
	if score > best_run[0]:
		best_run = [score, C]

print
svm_score, svm_C = best_run
print "Best overall: C=%5g with score=%.5f" % (svm_C, svm_score)

svm_classifier = SVC(C = svm_C, random_state = 0)
svm_classifier.fit(X[features], Y)

output_predictions(svm_classifier, "06_submission_svm.csv", formatting, features)

# Arbre "modèle 3"
tree_classifier = DecisionTreeClassifier(max_depth=6, min_samples_leaf=6, random_state = 0)
tree_classifier.fit(X[features], Y)

# Random Forest 200 arbres
forest_classifier = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=6, random_state = 0)
forest_classifier.fit(X[features], Y)

# Vote
class VoteClassifier(object):
	def __init__(self, classifiers):
		self.classifiers = classifiers

	def predict(self, X):
		predictions = np.zeros((X.shape[0], len(self.classifiers)))
		for idx, clf in enumerate(self.classifiers):
			predictions[:, idx] = clf.predict(X)
		return sp.stats.mode(predictions, axis=1)[0]

vote_classifier = VoteClassifier([svm_classifier, tree_classifier, forest_classifier])

output_predictions(vote_classifier, "06_submission_vote.csv", formatting, features)

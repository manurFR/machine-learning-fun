#!/usr/bin/env python
# -*- coding: UTF8 -*-

from pima_utils import load_and_prepare_pima, COLUMN_NAMES
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn import linear_model, grid_search, metrics
import numpy as np
import pylab as pl

train, test, train_target, test_target = load_and_prepare_pima()

print "## Perceptron ##\n"

print "Grid Search for alpha and n_iter"

cv=StratifiedKFold(train_target, n_folds=5)
perceptron = linear_model.Perceptron(random_state=0)
gridsearch = grid_search.GridSearchCV(perceptron, {'alpha': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
												   'n_iter': [5, 10, 15, 20, 50]},
									  cv=cv, scoring='f1')

gridsearch.fit(train, train_target)

print "Best params: {}".format(gridsearch.best_params_)
print "Best f1 score: %.5f" % gridsearch.best_score_
print

perceptron = linear_model.Perceptron(n_iter=15, random_state=0)
scores = cross_val_score(perceptron, train, train_target, scoring='f1', cv=cv)
print "Cross-validation (mean) f1-score: %.5f\n" % np.mean(scores)

perceptron.fit(train, train_target)
predicted = perceptron.predict(test)

print "Test accuracy score (with trained model): %.5f" % perceptron.score(test, test_target)
print metrics.classification_report(test_target, predicted, target_names=['non diabetic', 'diabetic'])

print "f1-score: %.5f" % metrics.f1_score(test_target, predicted)
print

print "intercept: %.5f" % perceptron.intercept_
print "coefs: "
print COLUMN_NAMES[:-1]
print perceptron.coef_

print '\nFirst 30 women of the test set:'
print 'known:     ' + str(test_target[:30])
print 'predicted: ' + str(predicted[:30])
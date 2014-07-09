#!/usr/bin/env python
# -*- coding: UTF8 -*-

from pima_utils import load_and_prepare_pima, COLUMN_NAMES
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn import neighbors, grid_search, metrics
import numpy as np

train, test, train_target, test_target = load_and_prepare_pima()

print "## K-Nearest Neighbors ##\n"
cv = StratifiedKFold(train_target, n_folds=5)

print "Grid Search for n_neighbors and weights"

knn = neighbors.KNeighborsClassifier()
gridsearch = grid_search.GridSearchCV(knn, {'n_neighbors': [3, 5, 10, 15, 20],
											'weights': ['uniform', 'distance']},
									  cv=cv, scoring='f1')

gridsearch.fit(train, train_target)

print "Best params: {}".format(gridsearch.best_params_)
print "Best f1 score: %.5f" % gridsearch.best_score_
print

knn = neighbors.KNeighborsClassifier(**gridsearch.best_params_)
scores = cross_val_score(knn, train, train_target, scoring='f1', cv=cv)
print "Cross-validation (mean) f1-score: %.5f\n" % np.mean(scores)

knn.fit(train, train_target)
predicted = knn.predict(test)

print "Test accuracy score (with trained model): %.5f" % knn.score(test, test_target)
print metrics.classification_report(test_target, predicted, target_names=['non diabetic', 'diabetic'])

print "f1-score: %.5f" % metrics.f1_score(test_target, predicted)
print

print '\nFirst 30 women of the test set:'
print 'known:     ' + str(test_target[:30])
print 'predicted: ' + str(predicted[:30])
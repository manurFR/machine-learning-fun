#!/usr/bin/env python
# -*- coding: UTF8 -*-

from pima_utils import load_and_prepare_pima, COLUMN_NAMES
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn import grid_search, metrics
import numpy as np
from sklearn.svm import SVC

train, test, train_target, test_target = load_and_prepare_pima()

print "## Support Vector Machines ##\n"
cv = StratifiedKFold(train_target, n_folds=5)

print "Grid Search for C, kernel, degree and gamma"

svm = SVC(random_state=0)
C_values = [0.1, 0.3, 1, 3, 10, 30]
grid = [ {'kernel': ['linear'], 'C': C_values},
		 {'kernel': ['rbf'],    'C': C_values, 'gamma': 10.0 ** np.arange(-5, 4)} ]

gridsearch = grid_search.GridSearchCV(svm, grid, cv=cv, scoring='f1', verbose=1)

gridsearch.fit(train, train_target)

print "Best params: {}".format(gridsearch.best_params_)
print "Best f1 score: %.5f" % gridsearch.best_score_
print

model = SVC(random_state=0, **gridsearch.best_params_)
scores = cross_val_score(model, train, train_target, scoring='f1', cv=cv)
print "Cross-validation (mean) f1-score: %.5f\n" % np.mean(scores)

model.fit(train, train_target)
predicted = model.predict(test)

print "Test accuracy score (with trained model): %.5f" % model.score(test, test_target)
print metrics.classification_report(test_target, predicted, target_names=['non diabetic', 'diabetic'])

print "f1-score: %.5f" % metrics.f1_score(test_target, predicted)
print

if model.kernel == 'linear':
	print "intercept: %.5f" % model.intercept_
	print "coefs: "
	print COLUMN_NAMES[:-1]
	print model.coef_

print '\nFirst 30 women of the test set:'
print 'known:     ' + str(test_target[:30])
print 'predicted: ' + str(predicted[:30])
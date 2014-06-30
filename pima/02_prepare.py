#!/usr/bin/env python
# -*- coding: UTF8 -*-

from pima_utils import load_pima, COLUMN_NAMES
from sklearn.cross_validation import train_test_split, StratifiedKFold, cross_val_score
from sklearn import preprocessing, linear_model, grid_search, metrics
import numpy as np

pima = load_pima()

target = pima.pop('diabetic')

# replace 0s by the mean of their feature, and scale each feature
pima['bmi'].replace(0, np.nan, inplace=True)
pima['blood pressure'].replace(0, np.nan, inplace=True)
pima['skin'].replace(0, np.nan, inplace=True)

imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
pima_prepared = preprocessing.scale(imputer.fit_transform(pima))

# split 75/25 between a train set and a test set
train, test, train_target, test_target = train_test_split(pima_prepared, target, train_size=0.75, random_state=0)

## Logistic Regression ##
print "## Logistic Regression ##"

# grid search for best C
print "Grid Search for best C"

linreg = linear_model.LogisticRegression()
gridsearch = grid_search.GridSearchCV(linreg, {'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]},
									  cv=StratifiedKFold(train_target, n_folds=5),
									  scoring='f1')

gridsearch.fit(train, train_target)

print "Best C: {}".format(gridsearch.best_params_)
print "Best score: %.5f" % gridsearch.best_score_
print

linreg = linear_model.LogisticRegression(C=gridsearch.best_params_['C'], random_state=0)
scores = cross_val_score(linreg, train, train_target, scoring='f1', cv=StratifiedKFold(train_target, n_folds=5))
print "Cross-validation (mean) f1-score: %.5f\n" % np.mean(scores)

linreg.fit(train, train_target)
predicted = linreg.predict(test)

print "Test accuracy score (with trained model): %.5f" % linreg.score(test, test_target)
print metrics.classification_report(test_target, predicted, target_names=['non diabetic', 'diabetic'])

print "tp: %d" % np.sum((test_target==1) * (predicted==1))
print "fp: %d" % np.sum((test_target==0) * (predicted==1))
print "fn: %d" % np.sum((test_target==1) * (predicted==0))
print "tn: %d" % np.sum((test_target==0) * (predicted==0))


print "f1-score: %.5f" % metrics.f1_score(test_target, predicted)
print

print "intercept: %.5f" % linreg.intercept_
print "coefs: "
print COLUMN_NAMES[:-1]
print linreg.coef_

print '\nFirst 30 women of the test set:'
print 'known:     ' + str(test_target[:30])
print 'predicted: ' + str(predicted[:30])

# TODO ajuster le seuil pour trouver le meilleur score f1
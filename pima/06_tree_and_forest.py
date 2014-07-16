#!/usr/bin/env python
# -*- coding: UTF8 -*-

from pima_utils import load_and_prepare_pima, COLUMN_NAMES
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn import grid_search, metrics
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz

train, test, train_target, test_target = load_and_prepare_pima()

print "## Decision Treee ##\n"
cv = StratifiedKFold(train_target, n_folds=5)

print "Grid Search for criterion, max_depth, min_samples_split and min_samples_leaf"

tree = DecisionTreeClassifier(random_state=0)
grid = [{'criterion': ['gini', 'entropy'], 
		 'max_depth': [3, 5, 8, 10, None],
		 'min_samples_split': [2, 5, 10, 20],
		 'min_samples_leaf':  [1, 2, 3, 5, 10]}]
gridsearch = grid_search.GridSearchCV(tree, grid, cv=cv, scoring='f1', verbose=1)

gridsearch.fit(train, train_target)

print "Best params: {}".format(gridsearch.best_params_)
print "Best f1 score: %.5f" % gridsearch.best_score_
print

model = DecisionTreeClassifier(random_state=0, **gridsearch.best_params_)
scores = cross_val_score(model, train, train_target, scoring='f1', cv=cv)
print "Cross-validation (mean) f1-score: %.5f\n" % np.mean(scores)

model.fit(train, train_target)
predicted = model.predict(test)

print "Test accuracy score (with trained model): %.5f" % model.score(test, test_target)
print metrics.classification_report(test_target, predicted, target_names=['non diabetic', 'diabetic'])

print "f1-score: %.5f" % metrics.f1_score(test_target, predicted)
print

print "feature importances: "
print COLUMN_NAMES[:-1]
print model.feature_importances_

print '\nFirst 30 women of the test set:'
print 'known:     ' + str(test_target[:30])
print 'predicted: ' + str(predicted[:30])

with open('decisiontree.dot', 'w') as f:
	f = export_graphviz(model, out_file = f, feature_names = COLUMN_NAMES[:-1])

print '\nGraphviz file exported'
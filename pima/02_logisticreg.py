#!/usr/bin/env python
# -*- coding: UTF8 -*-

from pima_utils import load_and_prepare_pima, COLUMN_NAMES
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn import linear_model, grid_search, metrics
import numpy as np
import pylab as pl

train, test, train_target, test_target = load_and_prepare_pima()

## Logistic Regression ##
print "## Logistic Regression ##"

# grid search for best C
print "Grid Search for best C"

linreg = linear_model.LogisticRegression()
gridsearch = grid_search.GridSearchCV(linreg, {'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]},
									  cv=StratifiedKFold(train_target, n_folds=5), scoring='f1')

gridsearch.fit(train, train_target)

print "Best C: {}".format(gridsearch.best_params_)
print "Best f1 score: %.5f" % gridsearch.best_score_
print

linreg = linear_model.LogisticRegression(C=gridsearch.best_params_['C'], random_state=0)
scores = cross_val_score(linreg, train, train_target, scoring='f1', cv=StratifiedKFold(train_target, n_folds=5))
print "Cross-validation (mean) f1-score: %.5f\n" % np.mean(scores)

linreg.fit(train, train_target)
predicted = linreg.predict(test)

print "Test accuracy score (with trained model): %.5f" % linreg.score(test, test_target)
print metrics.classification_report(test_target, predicted, target_names=['non diabetic', 'diabetic'])

print "tp: %d" % np.sum(test_target[predicted==1]==1)
print "fp: %d" % np.sum(test_target[predicted==1]==0)
print "fn: %d" % np.sum(test_target[predicted==0]==1)
print "tn: %d" % np.sum(test_target[predicted==0]==0)

print "f1-score: %.5f" % metrics.f1_score(test_target, predicted)
print

print "intercept: %.5f" % linreg.intercept_
print "coefs: "
print COLUMN_NAMES[:-1]
print linreg.coef_

print '\nFirst 30 women of the test set:'
print 'known:     ' + str(test_target[:30])
print 'predicted: ' + str(predicted[:30])

# Look for a better threshold ?
probas = linreg.predict_proba(train)
precision, recall, thresholds = metrics.precision_recall_curve(train_target, probas[:,1])
f1score = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1])

tps = np.array([np.sum(train_target[probas[:,1]>=t]==1) for t in thresholds])
tns = np.array([np.sum(train_target[probas[:,1]<t]==0) for t in thresholds])
accuracies = (tps + tns) / float(len(train))

pl.clf()
pl.plot(recall, precision, label='Precision-Recall curve')
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.ylim([0.0, 1.05])
pl.xlim([0.0, 1.0])
pl.title('Precision-Recall for Logistic Regression (Pima Indians)')
pl.legend(loc="lower left")
pl.show()

pl.clf()
pl.plot(thresholds, f1score, label='f1 score')
pl.plot(thresholds, accuracies, label='accuracy')
pl.plot(thresholds, precision[:-1], label='precision')
pl.plot(thresholds, recall[:-1], label='recall')
pl.xlabel('Thresholds')
pl.title('Metrics depending on threshold for Logistic Regression')
pl.legend(loc="lower left")
pl.show()

# New threshold found
selected_threshold = 0.45

print
print "Using the logistic regression with a more appropriate threshold: {}".format(selected_threshold)

test_probas = linreg.predict_proba(test)[:, 1]

predicted_new = test_probas >= selected_threshold

tp = np.sum(test_target[predicted_new]==1)
tn = np.sum(test_target[~predicted_new]==0)
print "Test accuracy score: %.5f" %  + ((tp + tn) / float(len(test)))

print metrics.classification_report(test_target, predicted_new, target_names=['non diabetic', 'diabetic'])

print "tp: %d" % np.sum(test_target[predicted_new]==1)
print "fp: %d" % np.sum(test_target[predicted_new]==0)
print "fn: %d" % np.sum(test_target[~predicted_new]==1)
print "tn: %d" % np.sum(test_target[~predicted_new]==0)

print "f1-score: %.5f" % metrics.f1_score(test_target, predicted_new)
print

print '\nFirst 30 women of the test set:'
print 'known:     ' + str(test_target[:30])
print 'predicted: ' + str(predicted_new[:30].astype(int))

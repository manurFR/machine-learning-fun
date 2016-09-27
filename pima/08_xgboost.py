import numpy as np
import xgboost
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.grid_search import GridSearchCV

from pima_utils import load_and_prepare_pima, COLUMN_NAMES

train, test, train_target, test_target = load_and_prepare_pima()

model = xgboost.XGBClassifier()
model.fit(train, train_target)

print "Test accuracy score (with trained model): %.5f" % model.score(test, test_target)

cv = StratifiedKFold(train_target, n_folds=5)
scores = cross_val_score(model, train, train_target, scoring='f1', cv=cv)
print "Cross-validation (mean) f1-score: %.5f\n" % np.mean(scores)

print "feature importances: "
print COLUMN_NAMES[:-1]
print model.feature_importances_
print

# hyperparameter tuning
print "Hyperparameter Tuning"
param_grid = dict(max_depth=[2, 4, 6, 8], n_estimators=[50, 100, 150], learning_rate=[0.001, 0.01, 0.1, 0.2, 0.3])
grid_search = GridSearchCV(model, param_grid, scoring="f1", cv=cv, verbose=1)
result = grid_search.fit(train, train_target)

print("Best: %f using %s" % (result.best_score_, result.best_params_))

best = xgboost.XGBClassifier(**grid_search.best_params_)
best.fit(train, train_target)

print "Test accuracy score (with trained model): %.5f" % best.score(test, test_target)

scores = cross_val_score(best, train, train_target, scoring='f1', cv=cv)
print "Cross-validation (mean) f1-score: %.5f\n" % np.mean(scores)

print "feature importances: "
print COLUMN_NAMES[:-1]
print best.feature_importances_
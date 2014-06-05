#!/usr/bin/env python
# -*- coding: UTF8 -*-

import numpy as np
import pandas as pan

from utils import load_train_data, plot_bias_variance
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

df, X, Y, median_ages = load_train_data()

datasizes = np.arange(50, len(X), 30)

train_errors = []
cv_errors = []
for datasize in datasizes:
	X_train, X_cv, y_train, y_cv = train_test_split(X[:datasize], Y[:datasize], test_size=0.7)
	classifier = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=8)
	classifier.fit(X_train,y_train)
	train_errors.append(1 - classifier.score(X_train, y_train))
	cv_errors.append(1 - classifier.score(X_cv, y_cv))

plot_bias_variance(datasizes, train_errors, cv_errors, "Learning curve (Bias/Variance detection) for Random Forest")	
# A small gap between train errors and cv errors stays even for large data size : maybe a bit of overfitting
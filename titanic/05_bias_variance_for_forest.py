#!/usr/bin/env python
# -*- coding: UTF8 -*-

import numpy as np
import pandas as pan
from matplotlib import pylab
from utils import load_train_data
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

def plot_bias_variance(data_sizes, train_errors, test_errors, title):
    pylab.figure(num=None, figsize=(6, 5))
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('Data set size')
    pylab.ylabel('Error')
    pylab.title(title)
    pylab.plot(
        data_sizes, test_errors, "--", data_sizes, train_errors, "b-", lw=1)
    pylab.legend(["train error", "test error"], loc="upper right")
    pylab.grid(True, linestyle='-', color='0.75')
    pylab.show()

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
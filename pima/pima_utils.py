#!/usr/bin/env python
# -*- coding: UTF8 -*-

import pandas as pan
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

PIMA_FILE = 'pima-indians-diabetes.data'
COLUMN_NAMES = ['pregnancies', 'glucose', 'blood pressure', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'diabetic']

def load_pima():
	return pan.read_csv(PIMA_FILE, names = COLUMN_NAMES)

def load_and_prepare_pima():
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

	return train, test, train_target, test_target	
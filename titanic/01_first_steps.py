#!/usr/bin/env python
# -*- coding: UTF8 -*-

import csv
import numpy as np
import scipy as sp


train = []
with open('train.csv', 'r') as f:
	for row in csv.reader(f):
		train.append(row)

headers = train[0]
train = np.array(train[1:])

# survival status
Y = train[:, 1]

# features (Pclass,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked)
X = train[:, (2, 4, 5, 6, 7, 8, 9, 10, 11)]
m, n = X.shape # m: nb of training examples, n: nb of features (n should be 9)

# split by sex
X_male = X[X[:,1] == 'male']
Y_male = Y[X[:,1] == 'male']
X_female = X[X[:,1] == 'female']
Y_female = Y[X[:,1] == 'female']

male_surviving_rate = 100 * float(np.sum(Y_male=='1')) / Y_male.size
female_surviving_rate = 100 * float(np.sum(Y_female=='1')) / Y_female.size
print "Surviving rate for men: %.2f%% and women: %.2f%%" % (male_surviving_rate, female_surviving_rate)
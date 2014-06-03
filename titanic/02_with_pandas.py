#!/usr/bin/env python
# -*- coding: UTF8 -*-

import csv, sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pan


df = pan.read_csv('train.csv', header=0)

# survival status
Y = df['Survived']

X = df[['Sex', 'Pclass', 'Fare', 'Age']]

X['SexBit'] = X['Sex'].map({'female': 0, 'male':1}).astype(int)
X['PclassIdx'] = X['Pclass'].map(lambda x: int(x)-1).astype(int)

# for this model, set all fares > 39 to be equal to 39, so that they all easily go to the 4th bin (which is the only important information for now)
X['FareFlat'] = np.minimum(X['Fare'].astype(float), 39.0*np.ones_like(X['Fare']))

# Replace missing ages by median value (of age) for the sex and the pclass
median_ages = np.zeros((2,3))
for i in range(2):
	for j in range(3):
		median_ages[i,j] = X[(X['SexBit'] == i) & (X['PclassIdx'] == j)]['Age'].dropna().median()
print "Median ages:", median_ages

X['AgeFill'] = X['Age']
for i in range(2):
	for j in range(3):
		X.loc[(X.Age.isnull()) & (X.SexBit == i) & (X.PclassIdx == j), 'AgeFill'] = median_ages[i,j]
X['AgeIsNull'] = pan.isnull(X.Age).astype(int)

# Engineered features
X['FamilySize'] = df['SibSp'] + df['Parch']
X['Age*Class'] = X.AgeFill * X.Pclass

# Drop useless string columns
X = X.drop(['Sex'], axis=1)

# For scikit-learn, we should convert the pandas dataframe to a numpy ndarray
#X = X.values

# survival by sex/pclass/fare bin
def fare_bin(fare):
	fare = float(fare)
	if fare < 10:
		return 0
	elif fare < 20:
		return 1
	elif fare < 30:
		return 2
	else:
		return 3

# Pclass,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
print "Survival rates by sex, pclass and fare :"
nb_bins = 4
nb_classes = len(np.unique(X['PclassIdx']))
survival_table = np.zeros((2, nb_classes, nb_bins)) # 2 genders x 3 classes x 4 fare bins matrix of survival rates

for i in xrange(nb_classes):
	for j in xrange(nb_bins):
		survival_women = Y[(X['SexBit'] == 0) & (X['PclassIdx'] == i) & (X['FareFlat'].astype(float) >= j*10) & (X['FareFlat'].astype(float) < (j+1)*10)]
		survival_men =   Y[(X['SexBit'] == 1) & (X['PclassIdx'] == i) & (X['FareFlat'].astype(float) >= j*10) & (X['FareFlat'].astype(float) < (j+1)*10)]
		survival_table[0,i,j] = np.mean(survival_women.astype(float))
		survival_table[1,i,j] = np.mean(survival_men.astype(float))
survival_table[ survival_table != survival_table ] = 0.

for i in xrange(nb_classes):
	for j in xrange(nb_bins):
		print "table: %s - %d - %d : %.2f %%" % ('female', i+1, j*10, survival_table[0,i,j] * 100)
for i in xrange(nb_classes):
	for j in xrange(nb_bins):		
		print "table: %s - %d - %d : %.2f %%" % ('male', i+1, j*10, survival_table[1,i,j] * 100)

# Predict survival by survival_rate > 50%
def predict(sexbit, pclassidx, fare):
	try:
		bin = int(min(float(fare), 39)) / 10
	except ValueError:
		bin = 2 - int(pclassidx) # if no fare, choose bin depending on pclass
	return survival_table[sexbit, pclassidx, bin] >= 0.5

def score(dataframe, classes):
	nb_good_predictions = 0.0
	total_predictions = classes.size
	for idx, i in dataframe.iterrows():
		prediction = predict(i['SexBit'], i['PclassIdx'], i['Fare'])
		if prediction == (classes[idx] == 1):
			nb_good_predictions += 1
	return nb_good_predictions / total_predictions

print "Score on training set : %.4f" % score(X, Y)

test = pan.read_csv('test.csv', header=0)
test['SexBit'] = test['Sex'].map({'female': 0, 'male':1}).astype(int)
test['PclassIdx'] = test['Pclass'].map(lambda x: int(x)-1).astype(int)

with open('02_submission.csv', 'w') as f:
	f.write("PassengerId,Survived\n")
	for idx, passenger in test.iterrows():
		f.write(str(passenger['PassengerId']) + "," + str(int(predict(passenger['SexBit'], passenger['PclassIdx'], passenger['Fare']))) + "\n")

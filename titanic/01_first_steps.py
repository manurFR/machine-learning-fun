#!/usr/bin/env python
# -*- coding: UTF8 -*-

import csv, sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


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
print "Surviving rate for men: %.2f %% and women: %.2f %%" % (male_surviving_rate, female_surviving_rate)

plt.clf()
# plot survival depending on fare
for (desc, data, classes) in [('Men', X_male, Y_male), ('Women', X_female, Y_female)]:
	histo = np.histogram(data[:, 6].astype(float), bins=[0,10,20,30,1000], weights=(classes=='1').astype(int))
	histo_total = np.histogram(data[:, 6].astype(float), bins=[0,10,20,30,1000]) # count of total population of each bin
	survival_rate_by_fare = 100*histo[0].astype(float)/histo_total[0]

	print "Survival rates by fare for %s :" % desc
	print ' ; '.join(survival_rate_by_fare.astype(str).tolist())

	# plt.title("Surviving Titanic passengers by fare (%s)" % desc)
	# plt.xlabel("Fare")
	# plt.ylabel("Nb of survivors")

	# plt.bar([0, 10, 20, 30], survival_rate_by_fare)
	# plt.show()

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
nb_classes = len(np.unique(X[:,0]))
survival_table = np.zeros((2, nb_classes, nb_bins)) # 2 genders x 3 classes x 4 fare bins matrix of survival rates

# for this model, set all fares > 39 to be equal to 39, so that they all easily go to the 4th bin (which is the only important information for now)
X_female[X_female[:,6].astype(float) > 39, 6] = 39.0
X_male[X_male[:,6].astype(float) > 39, 6] = 39.0

for i in xrange(nb_classes):
	for j in xrange(nb_bins):
		survival_women = Y_female[ (X_female[:,0].astype(int) == i+1) & (X_female[:,6].astype(float) >= j*10) & (X_female[:,6].astype(float) < (j+1)*10) ]
		survival_men = Y_male[ (X_male[:,0].astype(int) == i+1) & (X_male[:,6].astype(float) >= j*10) & (X_male[:,6].astype(float) < (j+1)*10) ]
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
def predict(sex, pclass, fare):
	sex_bit = 0 if sex == 'female' else 1
	try:
		bin = int(min(float(fare), 39)) / 10
	except ValueError:
		print sex, pclass, fare
		bin = 3 - int(pclass) # if no fare, choose bin depending on pclass
	return survival_table[sex_bit, int(pclass)-1, bin] >= 0.5

def score(dataset, classes):
	nb_good_predictions = 0.0
	total_predictions = classes.shape[0]
	for idx, i in enumerate(dataset):
		prediction = predict(i[1], i[0], i[6])
		if prediction == (classes[idx] == '1'):
			nb_good_predictions += 1
	return nb_good_predictions / total_predictions

print "Score on training set : %.4f" % score(X, Y)

test = []
with open('test.csv', 'r') as f:
	for row in csv.reader(f):
		test.append(row)

test = np.array(test[1:])

with open('01_submission.csv', 'w') as f:
	f.write("PassengerId,Survived\n")
	for passenger in test:
		f.write(passenger[0] + "," + str(int(predict(passenger[3], passenger[1], passenger[8]))) + "\n")

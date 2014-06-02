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
		return 10
	elif fare < 30:
		return 20
	else:
		return 30

print "Survival rates by sex, pclass and fare :"
repart = {'male':   {'1': {},
					 '2': {},
					 '3': {}},
		  'female': {'1': {},
					 '2': {},
					 '3': {}}}
for idx, i in enumerate(X):
	subclass = repart[i[1]][i[0]]
	bin = fare_bin(i[6])
	if bin not in subclass:
		subclass[bin] = {'population': 1, 'surviving': 0}
	else:
		subclass[bin]['population'] += 1
	if Y[idx] == '1':
		subclass[bin]['surviving'] += 1

for sex in repart:
	for pclass in repart[sex]:
		for bin in repart[sex][pclass]:
			repart[sex][pclass][bin]['survival_rate'] = 100 * float(repart[sex][pclass][bin]['surviving']) \
																	 / repart[sex][pclass][bin]['population']
			print "%s - %s - %d : %.2f %%" % (sex, pclass, bin, repart[sex][pclass][bin]['survival_rate'])


# Predict survival by survival_rate > 50%
def predict(sex, pclass, fare):
	bin = fare_bin(fare)
	if bin in repart[sex][pclass]:
		return repart[sex][pclass][bin]['survival_rate'] > 50.0
	else:
		return False

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
		f.write(passenger[0] + "," + str(int(predict(passenger[3], passenger[1], passenger[8] or 0.0))) + "\n")

#!/usr/bin/env python
# -*- coding: UTF8 -*-

import matplotlib.pyplot as plt
import pandas as pan
from pandas.tools.plotting import scatter_matrix
from pima_utils import load_pima

pima = load_pima()

print "size of the dataset: %d women" % len(pima)
nb_diabetes = len(pima[pima['diabetic'] == 1])
print "number of women diagnosed with diabetes within five years: %d (%.2f%%)" % (nb_diabetes, 100.0 * float(nb_diabetes) / len(pima))

print "\nfeatures analysis:"
print pima.describe()
print

pan.options.display.mpl_style = 'default'

# boxplot distribution of features
pima.boxplot()

# histogram distribution of features
pima.hist()

# same histogram distribution, separated for non-diabetic and diabetic women
pima.groupby('diabetic').hist()

# evaluate feature covariance and density
scatter_matrix(pima, alpha=0.2, figsize=(6, 6), diagonal='kde')

# compare glucose test result for diabetic and non-diabetic women
pima.groupby('diabetic').glucose.hist(alpha=0.4)

raw_input("Type enter to quit")
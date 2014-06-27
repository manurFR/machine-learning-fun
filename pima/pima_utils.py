#!/usr/bin/env python
# -*- coding: UTF8 -*-

import pandas as pan

PIMA_FILE = 'pima-indians-diabetes.data'
COLUMN_NAMES = ['pregnancies', 'glucose', 'blood pressure', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'diabetic']

def load_pima():
	pima = pan.read_csv(PIMA_FILE, names = COLUMN_NAMES)
	return pima
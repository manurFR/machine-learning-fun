#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import random


def split_data(data, probability):
    results = [], []
    for row in data:
        results[0 if random.random() < probability else 1].append(row)
    return results


def train_test_split(inputs, outputs, test_pct):
    data = zip(inputs, outputs)
    train, test = split_data(data, 1 - test_pct)
    inputs_train, outputs_train = zip(*train)
    inputs_test, outputs_test = zip(*test)
    return inputs_train, inputs_test, outputs_train, outputs_test


def accuracy(true_positive, false_positive, false_negative, true_negative):
    return (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)


def precision(true_positive, false_positive, _, __):
    return true_positive / (true_positive + false_positive)


def recall(true_positive, _, false_negative, __):
    return true_positive / (true_positive + false_negative)


def f1_score(true_positive, false_positive, false_negative, true_negative):
    p = precision(true_positive, false_positive, false_negative, true_negative)
    r = recall(true_positive, false_positive, false_negative, true_negative)
    return 2 * p * r / (p + r)
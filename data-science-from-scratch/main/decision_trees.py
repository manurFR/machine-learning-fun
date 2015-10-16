#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from collections import Counter
import collections
import math


def entropy(probabilities):
    return sum(-p * math.log(p, 2) for p in probabilities if p)


def class_probabilities(labels):
    total = len(labels)
    return [count / total for count in Counter(labels).values()]


def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)


def partition_entropy(subsets):
    """subsets is a list of lists of labeled data.
       the full entropy is the sum of the entropy of each subset, weighted by the size of the subset"""
    total = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset) * (len(subset) / total) for subset in subsets)


def partition_by(inputs, attribute):
    """inputs is a list of pairs ({'key': 'attribute', ...}, label)
       this returns a dict of {'attribute': pair1, ...}"""
    groups = collections.defaultdict(list)
    for pair in inputs:
        key = pair[0][attribute]
        groups[key].append(pair)
    return groups


def partition_entropy_by(inputs, attribute):
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())


if __name__ == '__main__':
    data = [
        ({'level': 'Senior', 'lang': 'Java', 'tweets': 'no', 'phd': 'no'}, False),
        ({'level': 'Senior', 'lang': 'Java', 'tweets': 'no', 'phd': 'yes'}, False),
        ({'level': 'Mid', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'R', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'R', 'tweets': 'yes', 'phd': 'yes'}, False),
        ({'level': 'Mid', 'lang': 'R', 'tweets': 'yes', 'phd': 'yes'}, True),
        ({'level': 'Senior', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, False),
        ({'level': 'Senior', 'lang': 'R', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'Python', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Senior', 'lang': 'Python', 'tweets': 'yes', 'phd': 'yes'}, True),
        ({'level': 'Mid', 'lang': 'Python', 'tweets': 'no', 'phd': 'yes'}, True),
        ({'level': 'Mid', 'lang': 'Java', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'Python', 'tweets': 'no', 'phd': 'yes'}, False)
    ]

    for key in data[0][0].keys():
        print key, partition_entropy_by(data, key)
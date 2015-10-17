#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from collections import Counter
import collections
from functools import partial
import math
import pprint


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


def classify(tree, input):
    """in the tree, nodes are represented by a tuple (attribute, {'value1': subnode1, 'value2': subnode2, ...})
       the subnodes can be other nodes of the same format, or leaves equal to either True of False
       'value1', 'value2', etc. are the different values that the specified attribute can have"""
    # leaf
    if tree in [True, False]:
        return tree

    attribute, subtree = tree

    key = input.get(attribute)  # None if input is missing attribute
    if key not in subtree:
        key = None

    return classify(subtree[key], input)


def build_tree_id3(inputs, split_candidates=None):
    if split_candidates is None:  # at the first pass, take all attributes as possible split candidates
        split_candidates = inputs[0][0].keys()

    num_inputs = len(inputs)
    num_true = len([label for item, label in inputs if label])
    num_false = num_inputs - num_true

    if num_true == 0:  # no Trues ? return a False leaf
        return False
    if num_false == 0:  # no Falses ? return a True leaf
        return True

    if not split_candidates:  # no split candidates left : take the label present the most time
        return num_true >= num_false

    # split on best attribute (lowest entropy)
    best_attribute = min(split_candidates, key=partial(partition_entropy_by, inputs))

    partitions = partition_by(inputs, best_attribute)
    new_candidates = [c for c in split_candidates if c != best_attribute]

    # build the subtrees (recursively)
    subtrees = {attribute_value: build_tree_id3(subset, new_candidates)
                for attribute_value, subset in partitions.iteritems()}

    # default case
    subtrees[None] = num_true > num_false

    return best_attribute, subtrees


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

    for k in data[0][0].keys():
        print k, partition_entropy_by(data, k)

    tree = build_tree_id3(data)
    pprint.pprint(tree)

    def should_hire(classification):
        return "HIRE!" if classification else "DON'T!"

    print 'junior javaist without phd that tweets :', should_hire(classify(
        tree, {'level': 'Junior', 'lang': 'Java', 'tweets': 'yes', 'phd': 'no'}))
    print 'junior javaist with a phd that tweets :', should_hire(classify(
        tree, {'level': 'Junior', 'lang': 'Java', 'tweets': 'yes', 'phd': 'yes'}))
    print 'intern :', should_hire(classify(tree, {'level': 'Intern'}))
    print 'senior without other values :', should_hire(classify(tree, {'level': 'Senior'}))

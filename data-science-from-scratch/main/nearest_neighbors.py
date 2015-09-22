#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from collections import Counter
from linalg import distance


def majority_vote(labels):
    """we assume that labels are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values() if count == winner_count])

    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])  # in case of equality, try again without the farthest


def knn_classify(k, labeled_points, new_point):
    """each labeled point should be a pair (point, label), point being a vector"""
    # sort the labeled points by their distance to the new point, nearest to farthest
    by_distance = sorted(labeled_points, key=lambda (point, _): distance(point, new_point))

    # extract the labels from only the k points closest to new point
    k_nearest_labels = [label for _, label in by_distance[:k]]

    # vote
    return majority_vote(k_nearest_labels)
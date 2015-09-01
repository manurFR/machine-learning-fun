from __future__ import division
from collections import Counter


def mean(vector):
    if not vector:
        return 0
    return sum(vector) / len(vector)


def median(vector):
    if not vector:
        return 0
    sorted_vector = sorted(vector)
    if len(sorted_vector) % 2 == 0:
        index_after_median = len(sorted_vector) // 2
        return mean(sorted_vector[(index_after_median - 1):(index_after_median + 1)])
    else:
        return sorted_vector[len(sorted_vector) // 2]


def quantile(vector, percentile):
    """the value (stricty) less than which this percentile of the (sorted) data lies"""
    if percentile >= 1.0:
        return max(vector) + 1
    sorted_vector = sorted(vector)
    return sorted_vector[int(percentile * len(sorted_vector))]


def mode(vector):
    """the list of values that are the most common"""
    if not vector:
        return []
    counts = Counter(vector)
    _, max_count = counts.most_common(1)[0]
    return sorted([component for component, count in counts.iteritems() if count == max_count])
from __future__ import division
from collections import Counter
import math
from linalg import sum_of_squares, dot


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


def data_range(vector):
    if not vector:
        return 0
    return max(vector) - min(vector)


def de_mean(vector):
    """deviation mean=vector - mean(vector)"""
    mean_of_vector = mean(vector)
    return [component - mean_of_vector for component in vector]


def variance(vector):
    """variance=sum of the squares of each component of the deviation mean vector, divided by n-1 (n=size of the vector)"""
    if len(vector) <= 1:
        return 0
    return sum_of_squares(de_mean(vector)) / (len(vector) - 1)


def standard_deviation(vector):
    return math.sqrt(variance(vector))


def interquartile_range(vector):
    return quantile(vector, 0.75) - quantile(vector, 0.25)


def covariance(vector1, vector2):
    if len(vector1) <= 1:
        return 0
    return dot(de_mean(vector1), de_mean(vector2)) / (len(vector1) - 1)


def correlation(vector1, vector2):
    stddev1 = standard_deviation(vector1)
    stddev2 = standard_deviation(vector2)
    if stddev1 == 0 or stddev2 == 0:
        return 0
    return covariance(vector1, vector2) / stddev1 / stddev2


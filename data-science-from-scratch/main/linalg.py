from __future__ import division
import math


def vector_add(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError('Both vectors should have the same size')
    return [sum(item) for item in zip(vector1, vector2)]


def vector_substract(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError('Both vectors should have the same size')
    return [a1 - a2 for a1, a2 in zip(vector1, vector2)]


def vector_sum(vectors):
    return reduce(vector_add, vectors)


def scalar_multiply(scalar, vector):
    return [scalar * i for i in vector]


def vector_mean(vectors):
    return scalar_multiply(1/len(vectors), vector_sum(vectors))


def dot(vector1, vector2):
    """dot product = sum of the products of each component one by one"""
    return sum(a1 * a2 for a1, a2 in zip(vector1, vector2))


def sum_of_squares(vector):
    """sum of the squares of each component of the vector"""
    return dot(vector, vector)


def magnitude(vector):
    """length of the vector, ie square root of the sum of the squares of each component"""
    return math.sqrt(sum_of_squares(vector))


def distance(vector1, vector2):
    return magnitude(vector_substract(vector1, vector2))
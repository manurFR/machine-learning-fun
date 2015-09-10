#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import Counter
import math
import random
import matplotlib.pyplot as plt
from linalg import get_column, shape, make_matrix
from probability import inverse_normal_cdf
from statistics import correlation


def bucketize(point, bucket_size):
    return bucket_size * math.floor(point / bucket_size)


def make_histogram(points, bucket_size):
    return Counter(bucketize(point, bucket_size) for point in points)


def plot_histogram(points, bucket_size, title=""):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
    plt.title(title)
    plt.show()


def columns_correlation(matrix, i, j):
    return correlation(get_column(matrix, i), get_column(matrix, j))


def correlation_matrix(matrix):
    _, num_columns = shape(matrix)
    return make_matrix(num_columns, num_columns, lambda i, j: columns_correlation(matrix, i, j))

if __name__ == '__main__':
    def random_normal():
        return inverse_normal_cdf(random.random())

    random.seed(0)

    uniform = [200 * random.random() - 100 for _ in range(20000)]
    normal = [54 * random_normal() for _ in range(20000)]

    plot_histogram(uniform, 10, "Uniform histogram")
    plot_histogram(normal, 10, "Normal histogram")

    # two dimensions
    xs = [random_normal() for _ in range(1000)]
    ys1 = [x + random_normal() / 2 for x in xs]
    ys2 = [-x + random_normal() / 2 for x in xs]

    plot_histogram(ys1, 0.5, "ys1")
    plot_histogram(ys2, 0.5, "ys2")

    plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
    plt.scatter(xs, ys2, marker='.', color='red', label='ys2')
    plt.legend(loc=9)
    plt.title("Very different joint distributions")
    plt.show()

    print correlation(xs, ys1), correlation(xs, ys2)


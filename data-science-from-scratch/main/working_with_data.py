#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
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

    # scatterplot matrix
    # prepare data
    def make_row():
        v0 = random_normal()
        v1 = -5 * v0 + random_normal()  # negatively correlated to v0
        v2 = v0 + v1 + 5 * random_normal()  # positively correlated to both v0 and v1
        v3 = 6 if v2 > -2 else 0  # depends exclusively on v2
        return [v0, v1, v2, v3]
    data = [make_row() for _ in range(100)]

    # plot it
    _, num_columns = shape(data)
    fig, ax = plt.subplots(num_columns, num_columns)

    for i in range(num_columns):
        for j in range(num_columns):
            if i != j:
                ax[i][j].scatter(get_column(data, j), get_column(data, i))
            else:
                ax[i][j].annotate("series " + str(i), (0.5, 0.5), xycoords='axes fraction', ha='center', va='center')

            # hide axis labels except for left and bottom charts
            if i < num_columns - 1:
                ax[i][j].xaxis.set_visible(False)
            if j > 0:
                ax[i][j].yaxis.set_visible(False)

    ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
    ax[0][0].set_ylim(ax[0][1].get_ylim())

    plt.show()



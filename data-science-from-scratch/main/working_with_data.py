#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from StringIO import StringIO
from collections import Counter, defaultdict
import csv
import math
import random
import dateutil.parser
import matplotlib.pyplot as plt
import requests
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


def parse_row(row, parsers):
    return [try_or_none(parser)(value) if parser is not None else value for value, parser in zip(row, parsers)]


def parse_rows_with(reader, parsers):
    for row in reader:
        yield parse_row(row, parsers)


def try_or_none(func):
    """wraps func to return None when it would raise an exception"""
    # noinspection PyBroadException
    def func_or_none(x):
        try:
            return func(x)
        except:
            return None
    return func_or_none


def try_parse_field(field_name, value, parser_dict):
    parser = parser_dict.get(field_name)
    if parser is not None:
        return try_or_none(parser)(value)
    else:
        return value


def parse_dict(input_dict, parser_dict):
    return {field_name: try_parse_field(field_name, value, parser_dict) for field_name, value in input_dict.iteritems()}


def picker(field_name):
    """returns a function that picks a field out of a dict"""
    return lambda d: d[field_name]


def pluck(field_name, rows):
    """from a list of dicts (rows), extract the value of a field name into a list"""
    return map(picker(field_name), rows)


def group_by(grouper_func, rows, value_transform=None):
    grouped_dict = defaultdict(list)
    for row in rows:
        grouped_dict[grouper_func(row)].append(row)

    if value_transform is None:
        return grouped_dict
    else:
        return {key: value_transform(rows) for key, rows in grouped_dict.iteritems()}

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

    # parsing data
    data = []

    stocks = "https://raw.githubusercontent.com/joelgrus/data-science-from-scratch/master/code/comma_delimited_stock_prices.csv"
    reader = csv.reader(StringIO(requests.get(stocks).text))
    for line in parse_rows_with(reader, [dateutil.parser.parse, None, float]):
        data.append(line)

    for row in data:
        if any(x is None for x in row):
            print row

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from StringIO import StringIO
from collections import Counter, defaultdict
import csv
from functools import partial
import math
import random
import dateutil.parser
import matplotlib.pyplot as plt
import requests
from gradient_descent import maximize_batch, maximize_stochastic
from linalg import get_column, shape, make_matrix, magnitude, dot, vector_sum
from probability import inverse_normal_cdf
from statistics import correlation, mean, standard_deviation


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


def scale(matrix):
    """returns the means and standard deviations of each column"""
    num_rows, num_columns = shape(matrix)
    means = [mean(get_column(matrix, j)) for j in range(num_columns)]
    stdevs = [standard_deviation(get_column(matrix, j)) for j in range(num_columns)]
    return means, stdevs


def rescale(matrix):
    """rescale the matrix so that each column has mean 0 and standard deviation 1"""
    means, stdevs = scale(matrix)

    def rescaled(i, j):
        if stdevs[j] > 0:
            return (matrix[i][j] - means[j]) / stdevs[j]
        else:
            return matrix[i][j]

    num_rows, num_columns = shape(matrix)
    return make_matrix(num_rows, num_columns, rescaled)


def de_mean_matrix(matrix):
    """make each column have mean 0 by centering each element to its former mean"""
    nr, nc = shape(matrix)
    means, _ = scale(matrix)
    return make_matrix(nr, nc, lambda i, j: matrix[i][j] - means[j])


def direction(vector):
    """rescale the vector to have length (magnitude) 1"""
    mag = magnitude(vector)
    return [component / mag for component in vector]


def directional_variance_row(row, vector):
    """the variance of the row in the direction determined by the vector"""
    return dot(row, direction(vector)) ** 2


def directional_variance(matrix, vector):
    return sum([directional_variance_row(row, vector) for row in matrix])


def directional_variance_gradient_row(row, vector):
    """the contribution of this row to the gradient of the direction(vector) variance"""
    return [2 * component * dot(row, direction(vector)) for component in row]


def directional_variance_gradient(matrix, vector):
    return vector_sum(directional_variance_gradient_row(row, vector) for row in matrix)


def first_principal_component(matrix):
    guess = [1 for _ in matrix[0]]
    unscaled_maximizer = maximize_batch(
        partial(directional_variance, matrix),
        partial(directional_variance_gradient, matrix),
        guess
    )
    return direction(unscaled_maximizer)


def first_principal_component_stochastic(matrix):
    guess = [1 for _ in matrix[0]]
    unscaled_maximizer = maximize_stochastic(
        lambda x, _, vector: directional_variance_row(x, vector),
        lambda x, _, vector: directional_variance_gradient_row(x, vector),
        matrix,
        [None for _ in matrix[0]],  # fake "y"
        guess
    )
    return direction(unscaled_maximizer)

if __name__ == '__main__':
    # def random_normal():
    #     return inverse_normal_cdf(random.random())
    #
    # random.seed(0)
    #
    # uniform = [200 * random.random() - 100 for _ in range(20000)]
    # normal = [54 * random_normal() for _ in range(20000)]
    #
    # plot_histogram(uniform, 10, "Uniform histogram")
    # plot_histogram(normal, 10, "Normal histogram")
    #
    # # two dimensions
    # xs = [random_normal() for _ in range(1000)]
    # ys1 = [x + random_normal() / 2 for x in xs]
    # ys2 = [-x + random_normal() / 2 for x in xs]
    #
    # plot_histogram(ys1, 0.5, "ys1")
    # plot_histogram(ys2, 0.5, "ys2")
    #
    # plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
    # plt.scatter(xs, ys2, marker='.', color='red', label='ys2')
    # plt.legend(loc=9)
    # plt.title("Very different joint distributions")
    # plt.show()
    #
    # print correlation(xs, ys1), correlation(xs, ys2)
    #
    # # scatterplot matrix
    # # prepare data
    # def make_row():
    #     v0 = random_normal()
    #     v1 = -5 * v0 + random_normal()  # negatively correlated to v0
    #     v2 = v0 + v1 + 5 * random_normal()  # positively correlated to both v0 and v1
    #     v3 = 6 if v2 > -2 else 0  # depends exclusively on v2
    #     return [v0, v1, v2, v3]
    # data = [make_row() for _ in range(100)]
    #
    # # plot it
    # _, num_columns = shape(data)
    # fig, ax = plt.subplots(num_columns, num_columns)
    #
    # for i in range(num_columns):
    #     for j in range(num_columns):
    #         if i != j:
    #             ax[i][j].scatter(get_column(data, j), get_column(data, i))
    #         else:
    #             ax[i][j].annotate("series " + str(i), (0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
    #
    #         # hide axis labels except for left and bottom charts
    #         if i < num_columns - 1:
    #             ax[i][j].xaxis.set_visible(False)
    #         if j > 0:
    #             ax[i][j].yaxis.set_visible(False)
    #
    # ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
    # ax[0][0].set_ylim(ax[0][1].get_ylim())
    #
    # plt.show()

    # parsing data
    data = []

    stocks = "https://raw.githubusercontent.com/joelgrus/data-science-from-scratch/master/code/comma_delimited_stock_prices.csv"
    reader = csv.reader(StringIO(requests.get(stocks).text))
    for line in parse_rows_with(reader, [dateutil.parser.parse, None, float]):
        data.append(line)

    for row in data:
        if any(x is None for x in row):
            print row

    dictdata = [{'date': row[0], 'symbol': row[1], 'closing_price': row[2]} for row in data]

    max_price_by_symbol = group_by(picker("symbol"),
                                   dictdata,
                                   lambda rows: max(pluck("closing_price", rows)))
    print max_price_by_symbol

    # more complicated grouping
    fullstocks = "https://raw.githubusercontent.com/joelgrus/data-science-from-scratch/master/code/stocks.txt"
    reader = csv.DictReader(StringIO(requests.get(fullstocks).text), delimiter="\t")
    stocksdata = [parse_dict(row, {'date': dateutil.parser.parse, 'closing_price': float}) for row in reader]

    def percent_price_change(yesterday, today):
        return today["closing_price"] / yesterday["closing_price"] - 1

    def day_over_day_changes(grouped_rows):
        ordered = sorted(grouped_rows, key=picker("date"))

        return [{"symbol": today["symbol"], "date": today["date"], "change": percent_price_change(yesterday, today)}
                for yesterday, today in zip(ordered, ordered[1:])]

    changes_by_symbol = group_by(picker("symbol"), stocksdata, day_over_day_changes)

    all_changes = [change for changes in changes_by_symbol.values() for change in changes]

    print max(all_changes, key=picker("change")), min(all_changes, key=picker("change"))

    def combine_pct_changes(pct_change1, pct_change2):
        """computes the cumulative effect of two percentage changes, eg +10% -20% yields -12% (and not -10%)"""
        return (1 + pct_change1) * (1 + pct_change2) - 1

    def overall_changes(changes):
        return reduce(combine_pct_changes, pluck("change", changes))

    overall_change_by_month = group_by(lambda row: row['date'].month, all_changes, overall_changes)
    print overall_change_by_month

    # rescaling
    data = [[1, 20, 2],
            [1, 30, 3],
            [1, 40, 4]]

    print "original: ", data
    print "scale: ", scale(data)
    print "rescaled: ", rescale(data)

    # dimensionality reduction
    X = [
        [20.9666776351559,-13.1138080189357],
        [22.7719907680008,-19.8890894944696],
        [25.6687103160153,-11.9956004517219],
        [18.0019794950564,-18.1989191165133],
        [21.3967402102156,-10.8893126308196],
        [0.443696899177716,-19.7221132386308],
        [29.9198322142127,-14.0958668502427],
        [19.0805843080126,-13.7888747608312],
        [16.4685063521314,-11.2612927034291],
        [21.4597664701884,-12.4740034586705],
        [3.87655283720532,-17.575162461771],
        [34.5713920556787,-10.705185165378],
        [13.3732115747722,-16.7270274494424],
        [20.7281704141919,-8.81165591556553],
        [24.839851437942,-12.1240962157419],
        [20.3019544741252,-12.8725060780898],
        [21.9021426929599,-17.3225432396452],
        [23.2285885715486,-12.2676568419045],
        [28.5749111681851,-13.2616470619453],
        [29.2957424128701,-14.6299928678996],
        [15.2495527798625,-18.4649714274207],
        [26.5567257400476,-9.19794350561966],
        [30.1934232346361,-12.6272709845971],
        [36.8267446011057,-7.25409849336718],
        [32.157416823084,-10.4729534347553],
        [5.85964365291694,-22.6573731626132],
        [25.7426190674693,-14.8055803854566],
        [16.237602636139,-16.5920595763719],
        [14.7408608850568,-20.0537715298403],
        [6.85907008242544,-18.3965586884781],
        [26.5918329233128,-8.92664811750842],
        [-11.2216019958228,-27.0519081982856],
        [8.93593745011035,-20.8261235122575],
        [24.4481258671796,-18.0324012215159],
        [2.82048515404903,-22.4208457598703],
        [30.8803004755948,-11.455358009593],
        [15.4586738236098,-11.1242825084309],
        [28.5332537090494,-14.7898744423126],
        [40.4830293441052,-2.41946428697183],
        [15.7563759125684,-13.5771266003795],
        [19.3635588851727,-20.6224770470434],
        [13.4212840786467,-19.0238227375766],
        [7.77570680426702,-16.6385739839089],
        [21.4865983854408,-15.290799330002],
        [12.6392705930724,-23.6433305964301],
        [12.4746151388128,-17.9720169566614],
        [23.4572410437998,-14.602080545086],
        [13.6878189833565,-18.9687408182414],
        [15.4077465943441,-14.5352487124086],
        [20.3356581548895,-10.0883159703702],
        [20.7093833689359,-12.6939091236766],
        [11.1032293684441,-14.1383848928755],
        [17.5048321498308,-9.2338593361801],
        [16.3303688220188,-15.1054735529158],
        [26.6929062710726,-13.306030567991],
        [34.4985678099711,-9.86199941278607],
        [39.1374291499406,-10.5621430853401],
        [21.9088956482146,-9.95198845621849],
        [22.2367457578087,-17.2200123442707],
        [10.0032784145577,-19.3557700653426],
        [14.045833906665,-15.871937521131],
        [15.5640911917607,-18.3396956121887],
        [24.4771926581586,-14.8715313479137],
        [26.533415556629,-14.693883922494],
        [12.8722580202544,-21.2750596021509],
        [24.4768291376862,-15.9592080959207],
        [18.2230748567433,-14.6541444069985],
        [4.1902148367447,-20.6144032528762],
        [12.4332594022086,-16.6079789231489],
        [20.5483758651873,-18.8512560786321],
        [17.8180560451358,-12.5451990696752],
        [11.0071081078049,-20.3938092335862],
        [8.30560561422449,-22.9503944138682],
        [33.9857852657284,-4.8371294974382],
        [17.4376502239652,-14.5095976075022],
        [29.0379635148943,-14.8461553663227],
        [29.1344666599319,-7.70862921632672],
        [32.9730697624544,-15.5839178785654],
        [13.4211493998212,-20.150199857584],
        [11.380538260355,-12.8619410359766],
        [28.672631499186,-8.51866271785711],
        [16.4296061111902,-23.3326051279759],
        [25.7168371582585,-13.8899296143829],
        [13.3185154732595,-17.8959160024249],
        [3.60832478605376,-25.4023343597712],
        [39.5445949652652,-11.466377647931],
        [25.1693484426101,-12.2752652925707],
        [25.2884257196471,-7.06710309184533],
        [6.77665715793125,-22.3947299635571],
        [20.1844223778907,-16.0427471125407],
        [25.5506805272535,-9.33856532270204],
        [25.1495682602477,-7.17350567090738],
        [15.6978431006492,-17.5979197162642],
        [37.42780451491,-10.843637288504],
        [22.974620174842,-10.6171162611686],
        [34.6327117468934,-9.26182440487384],
        [34.7042513789061,-6.9630753351114],
        [15.6563953929008,-17.2196961218915],
        [25.2049825789225,-14.1592086208169]
    ]

    Y = de_mean_matrix(X)
    components = first_principal_component(Y)
    print components

    plt.scatter(get_column(Y, 0), get_column(Y, 1), marker='.', color='black')
    plt.quiver(components[0], components[1], color='red')
    # plt.legend(loc=9)
    plt.title("First principal component")
    plt.show()

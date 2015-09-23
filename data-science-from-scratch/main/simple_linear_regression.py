#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from statistics import correlation, standard_deviation, mean, de_mean


def predict(alpha, beta, x_i):
    return x_i * alpha + beta


def error(alpha, beta, x_i, y_i):
    return y_i - predict(alpha, beta, x_i)


def sum_of_squared_errors(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))


def least_squares_fit(x, y):
    """numerical 'perfect' determination of alpha, beta for linear regression"""
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta


def total_sum_of_squares(vector):
    return sum(m ** 2 for m in de_mean(vector))


def r_squared(alpha, beta, x, y):
    return 1.0 - (sum_of_squared_errors(alpha, beta, x, y) / total_sum_of_squares(y))


def squared_error(x_i, y_i, theta):
    alpha, beta = theta
    return error(alpha, beta, x_i, y_i) ** 2


def squared_error_gradient(x_i, y_i, theta):
    alpha, beta = theta
    return [-2 * error(alpha, beta, x_i, y_i),          # alpha partial derivative
            -2 * error(alpha, beta, x_i, y_i) * x_i]    # beta partial derivative


# example usage of gradient descent to approximate alpha, beta :
# theta = [...,  ...]  # a (random?) starting hypothesis
# alpha, beta = minimize_stochastic(squared_error,
#                                   squared_error_gradient,
#                                   x_values,
#                                   y_values,
#                                   theta,
#                                   0.0001)
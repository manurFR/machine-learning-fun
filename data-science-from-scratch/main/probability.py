#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import random
import matplotlib.pyplot as plt
import math

# pdf = probability density function ; the integral of this function over an interval is the probability of
#       of a random variable following the corresponding distribution to have a value in that interval
# cdf = cumulative distribution function : probability of a random variable following the corresponding distribution
#       to be less or equal than a certain value


def uniform_pdf(value):
    return 1 if 0 <= value < 1 else 0


def uniform_cdf(value):
    return min(1, max(0, value))


def normal_pdf(value, mu=0, sigma=1):
    """f(x|mu,sigma) = 1/(sigma * sqrt(2*PI)) * exp( -(x-mu)² / (2 * sigma²) )"""
    return math.exp((-1 * (value - mu)**2) / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))


def normal_cdf(value, mu=0, sigma=1):
    """F(x)  = (1 + erf( (x-mu) / (sigma * sqrt(2)) )) / 2, avec erf = 'error function'"""
    return (1 + math.erf((value - mu) / (sigma * math.sqrt(2)))) / 2


def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    """Given a probability p and a normal distribution of parameters (mu, sigma), gives the value from the normal
       cumulative distribution function which yields probability p with the given tolerance. We approximate by
        binary search"""

    # if not standard, compute standard and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z, low_p = -10.0, 0            # normal_cdf(-10) is (very close to) 0
    hi_z,  hi_p  =  10.0, 1            # normal_cdf(10)  is (very close to) 1
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2     # consider the midpoint
        mid_p = normal_cdf(mid_z)      # and the cdf's value there
        if mid_p < p:
            # midpoint is still too low, search above it
            low_z, low_p = mid_z, mid_p
        elif mid_p > p:
            # midpoint is still too high, search below it
            hi_z, hi_p = mid_z, mid_p
        else:
            break

    return mid_z


def bernouilli_trial(p):
    """The Bernoulli random variable equals 1 with probability p and 0 with probability 1-p.
       Its mean is p and standard deviation sqrt(p*(1-p))"""
    return 1 if random.random() < p else 0


def binomial(n, p):
    """The sum of n independant Bernoulli random variables of probability p.
       The Central Limit theorem states that as n gets large, the binomial variable approximates a random normal variable
       of mean n*p (mu) and standard deviation sqrt(n*p*(1-p)) (sigma)."""
    return sum(bernouilli_trial(p) for _ in range(n))


### when not used as lib
if __name__ == '__main__':
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs, [normal_pdf(x) for x in xs], '-', label='mu=0,sigma=1')
    plt.plot(xs, [normal_pdf(x, sigma=2) for x in xs], '--', label='mu=0,sigma=2')
    plt.plot(xs, [normal_pdf(x, mu=-1, sigma=1.5) for x in xs], ':', label='mu=-1,sigma=1.5')
    plt.legend()
    plt.title("Normal probability density functions")
    plt.show()

    plt.plot(xs, [normal_cdf(x) for x in xs], '-', label='mu=0,sigma=1')
    plt.plot(xs, [normal_cdf(x, sigma=2) for x in xs], '--', label='mu=0,sigma=2')
    plt.plot(xs, [normal_cdf(x, mu=-1, sigma=1.5) for x in xs], ':', label='mu=-1,sigma=1.5')
    plt.legend(loc=4)
    plt.title("Normal cumulative density functions")
    plt.show()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import math
from probability import normal_cdf, inverse_normal_cdf


# On part de n tirages indépendants et d'une hypothèse null (à confirmer/infirmer) qui affirme qu'un évènement
# se produit pour chaque tirage avec une probabilité p ; le nombre de tirages pour lesquels cet évènement se produit
# (que j'appelle ici les tirages positifs) est une fonction binomiale (n tirages indépendants de probabilité p).
# On calcule les paramètres mu, sigma de la loi normale qui approxime notre binomiale de paramètres n, p.
def normal_approximation_of_binomial(n, p):
    mu = p * n
    sigma = math.sqrt(p * n * (1 - p))
    return mu, sigma


# La probabilité que nos n tirages aient produit l'évènement attendu un nombre de fois inférieur ou égal à "value"
# est la fonction de répartition de la loi normale.
# Autrement dit: P(nb de tirages positifs <= value) = normal_cdf(value, mu, sigma)
normal_probability_below = normal_cdf


# P(nb de tirages positifs > value)
def normal_probability_above(lo, mu=0, sigma=1):
    return 1 - normal_probability_below(lo, mu, sigma)


# P(lo < nb de tirages positifs <= hi)
def normal_probability_between(lo, hi, mu=0, sigma=1):
    return normal_probability_below(hi, mu, sigma) - normal_probability_below(lo, mu, sigma)


# P(nb de tirages positifs <= lo ET > hi)
def normal_probability_outside(lo, hi, mu=0, sigma=1):
    return 1 - normal_probability_between(lo, hi, mu, sigma)


# Calcule le seuil pour lequel on aura la probabilité donnée d'avoir moins ou égal de tirages positifs
def normal_upper_bound(probability, mu=0, sigma=1):
    return inverse_normal_cdf(probability, mu, sigma)


# Le seuil pour lequel on aura la probabilité donnée d'avoir strictement plus de tirages positifs
def normal_lower_bound(probability, mu=0, sigma=1):
    return inverse_normal_cdf(1 - probability, mu, sigma)


# Les seuils symétriques entre lesquels on aura la probabilité donnée d'avoir le nombre de tirages positifs
def normal_two_sided_bounds(probability, mu=0, sigma=1):
    tail_probability = (1 - probability) / 2
    # upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)
    # lower bound should have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)
    return lower_bound, upper_bound

mu, sigma = normal_approximation_of_binomial(1000, 0.5)
print "normal parameters for n = 1000 times, p = 0.5 :", mu, sigma
print "probability for 1000 coin flips to bring between 484 (excluded) and 515 (included) heads :", normal_probability_between(484, 515, mu, sigma)
print "how many heads (for 1000 coin flips) can we expect to get with a 75% probability ?", int(normal_lower_bound(0.75, mu, sigma))
print "how many heads (for 1000 coin flips) can we expect to get with a 5% probability ?", int(normal_lower_bound(0.05, mu, sigma))
print "range of how many heads around the mean we can expect to get with a 67% probability ?", map(int, normal_two_sided_bounds(0.67, mu, sigma))

print
# notre hypothèse null sera statistiquement significative (significance) si la probabilité de rejeter l'hypothèse alors
#  qu'elle est juste est inférieure à 5%
print "intervalle du nb de face que l'on s'attend à avoir avec une probabilité de 95% :", map(int, normal_two_sided_bounds(0.95, mu, sigma))
print "=> si notre observation est dans cet intervalle 95% du temps, c'est que notre hypothèse de départ (p=0.5) est juste"
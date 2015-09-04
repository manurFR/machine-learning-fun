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
#  qu'elle est juste est inférieure à 5% (convention)
print "intervalle du nb de face que l'on s'attend à avoir avec une probabilité de 95% :", map(int, normal_two_sided_bounds(0.95, mu, sigma))
print "=> si notre observation est dans cet intervalle 95% du temps, c'est que notre hypothèse de départ (p=0.5) est juste"

# La puissance statistique d'un test est la probabilité de rejeter l'hypothèse null lorsque l'hypothèse alternative est vraie.
# power = P(reject Ho|H1 is true) = 1 - P(Ho is not rejected|H1 true)
# L'hypothèse null Ho est que p = 0.5.
# Une hypothèse alternative H1 peut être que p = 0.55 (pièce légèrement biaisée vers le côté face).

# On construit la distribution dans l'hypothèse où H1 serait vrai :
mu_h1, sigma_h1 = normal_approximation_of_binomial(1000, 0.55)
# Pour que Ho ne soit pas rejetée dans ce cas, il faut que le nb de tirages positifs respecte l'intervalle de
# significativité statistique à 95% de celle ci :
lo_h0, hi_h0 = normal_two_sided_bounds(0.95, mu, sigma)
# La probabilité P(Ho is not rejected|H1 true) est donc la probabilité que les tirages avec les paramètres de H1
# rentrent dans les bornes de significativité de Ho :
proba_h0_rejected_given_h1 = normal_probability_between(lo_h0, hi_h0, mu_h1, sigma_h1)
print "puissance de l'hypothèse H1(p=0.55) contre l'hypothèse null Ho(p=0.50) avec n=1000: ", 1 - proba_h0_rejected_given_h1
print "=> c'est à dire la probabilité de bien rejeter Ho si H1 est vrai"

print
# autre Ho : la pièce n'est pas biaisée côté face : p <= 0.5
# cette hypothèse est toujours statistiquement significative si P(reject Ho|Ho true) <= 5%, mais cette fois on l'exprime
# en cherchant le seuil pour lequel on aura 95% d'avoir moins ou égal de tirages positifs :
hi_h0 = normal_upper_bound(0.95, mu, sigma)
print hi_h0
# L'hypothèse concurrente H1 est que p > 0.5, sa puissance dépend de la probabilité P(Ho is not rejected|H1 true),
# donc à nouveau que les tirages avec les paramètres de H1 rentrent dans les bornes de significativité de Ho :
proba_h0_rejected_given_h1 = normal_probability_below(hi_h0, mu_h1, sigma_h1)
print "puissance de l'hypothèse H1(p>0.5) contre l'hypothèse null Ho(p<=0.5) avec n=1000: ", 1 - proba_h0_rejected_given_h1

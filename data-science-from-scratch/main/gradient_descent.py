#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import random
from linalg import distance


def difference_quotient(function, x, delta):
    return (function(x + delta) - function(x)) / delta


def partial_difference_quotient(function, coordinates, index_coordinate, delta):
    increment = [coordinate + (delta if idx == index_coordinate else 0) for idx, coordinate in enumerate(coordinates)]
    return (function(increment) - function(coordinates)) / delta


def estimate_gradient(function, coordinates, delta=0.00001):
    return [partial_difference_quotient(function, coordinates, index_coordinate, delta)
            for index_coordinate, _ in enumerate(coordinates)]


def step(coordinates, direction, step):
    return [coordinate + (direction_coordinate * step) for coordinate, direction_coordinate in zip(coordinates, direction)]


def safe(function):
    """returns a new function that return the same results as the input function, except when it produces errors,
       in which case it returns infinity. That way, erroneous inputs can never end up as minimums."""
    # noinspection PyBroadException
    def safe_function(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except:
            return float('inf')
    return safe_function


def minimize_batch(target_func, gradient_func, theta_0, tolerance=0.000001):
    """Use gradient descent to find theta that minimizes target function"""
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]  # candidates

    theta = theta_0
    target_func = safe(target_func)
    value = target_func(*theta)

    while True:
        gradient = gradient_func(theta)
        # calcule les prochains theta possibles pour chacun des step_size candidats
        next_thetas = [step(theta, gradient, -step_size) for step_size in step_sizes]
        # choisit le next_theta qui minimize la fonction à optimiser
        next_theta = min(next_thetas, key=target_func)
        next_value = target_func(next_theta)

        # est-ce qu'on converge ?
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value

if __name__ == '__main__':
    # gradient descent de la fonction 'carré' -- la dérivée de f(x)=x**2 est f'(x)=2x
    # Comme on peut la définir mathématiquement, on utilise directement cette dérivée dans l'algorithme, au lieu
    # de l'estimer par estimate_gradient() qui est très couteux en nb d'appels à la fonction que l'on cherche
    # à optimiser.
    def gradient_of_square(coordinates):
        return [2 * c for c in coordinates]

    point = [random.randint(-10, 10) for _ in range(3)]  # random starting point
    tolerance = 0.0000001
    while True:
        gradient = gradient_of_square(point)
        next_point = step(point, gradient, -0.01)  # step négative pour minimiser la fonction et chercher le minimum
        if distance(next_point, point) < tolerance:
            break
        point = next_point
    print "Minimum trouvé :", point


def negate(function):
    return lambda *args, **kwargs: -function(*args, **kwargs)


def negate_all(function):
    return lambda *args, **kwargs: [-res for res in function(*args, **kwargs)]


def maximize_batch(target_func, gradient_func, theta_0, tolerance=0.000001):
    return minimize_batch(negate(target_func),
                          negate_all(gradient_func),
                          theta_0,
                          tolerance)


def in_random_order(data):
    """generator that returns the elements of data in random order"""
    indexes = range(len(data))
    random.shuffle(indexes)
    for i in indexes:
        yield data[i]
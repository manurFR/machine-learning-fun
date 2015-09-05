#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import random
from unittest import TestCase
import math
import mock
import gradient_descent
from linalg import scalar_multiply, vector_add


class TestGradientDescent(TestCase):
    def test_difference_quotient(self):
        self.assertEqual(0, gradient_descent.difference_quotient(lambda x: 1, 2, 0.05))
        self.assertEqual((2.1 ** 2 - 4) / 0.1, gradient_descent.difference_quotient(lambda x: x ** 2, 2, 0.1))

    def test_partial_difference_quotient(self):
        self.assertEqual(0, gradient_descent.partial_difference_quotient(lambda v: v[0] + v[1], [1, 2, 3], 2, 0.1))
        self.assertEqual((1*2*3.1 - 1*2*3) / 0.1,
                         gradient_descent.partial_difference_quotient(lambda v: v[0]*v[1]*v[2], [1, 2, 3], 2, 0.1))

    def test_estimate_gradient(self):
        self.assertEqual([0, 0, 0], gradient_descent.estimate_gradient(lambda v: 1, [1, 2, 3], 0.1))
        self.assertEqual([(1.1*2*3 - 6) / 0.1, (1*2.1*3 - 6) / 0.1, (1*2*3.1 - 6) / 0.1],
                         gradient_descent.estimate_gradient(lambda v: v[0]*v[1]*v[2], [1, 2, 3], 0.1))

    def test_step(self):
        self.assertEqual([1.1, 1.9, 3], gradient_descent.step([1, 2, 3], [1, -1, 0], 0.1))

    def test_safe(self):
        really_safe = gradient_descent.safe(lambda x: 1/x)
        self.assertEqual(1/2, really_safe(2))
        self.assertTrue(math.isinf(really_safe(0)))

    def test_minimize_batch(self):
        optimized = gradient_descent.minimize_batch(
            lambda v: v[0] ** 2 + v[1] ** 2 + v[2] ** 2,
            lambda v: scalar_multiply(2, v),  # 2*[x y z] = [2x 2y 2z]
            [3, 2, 1],
            tolerance=0.000001
        )
        for index, value in enumerate(optimized):
            self.assertAlmostEquals(0, value, places=2,
                                    msg='Value {0} not optimized to 0 for dimension of index: {1}'.format(value, index))

    def test_negate(self):
        self.assertEqual(-3, gradient_descent.negate(lambda x: x+2)(1))
        self.assertEqual(1, gradient_descent.negate(lambda x: x)(-1))
        self.assertEqual(0, gradient_descent.negate(lambda x: x)(0))

    def test_negate_all(self):
        self.assertEqual([2, -5, 0], gradient_descent.negate_all(lambda v: [v-4, v+3, v-2])(2))

    def test_maximize_batch(self):
        # cette parabole est maximisée pour [-1 -1]
        optimized = gradient_descent.maximize_batch(
            lambda v: - ((v[0]+1) ** 2 + (v[1]+1) ** 2),  # f(x,y)= - ((x+1)**2 + (y+1)**2)
            lambda v: scalar_multiply(-2, vector_add(v, [1, 1])),  # f'(x,y) = [-2(x+1) -2(y+1)]
            [3, 2],
            tolerance=0.000001
        )
        for index, value in enumerate(optimized):
            self.assertAlmostEquals(-1, value, places=2,
                                    msg='Value {0} not optimized to -1 for dimension of index: {1}'.format(value, index))

    def test_in_random_order(self):
        def mock_shuffle(v):
            v[0], v[1], v[2], v[3] = v[3], v[0], v[2], v[1]
        with mock.patch.object(random, 'shuffle', side_effect=mock_shuffle):
            self.assertEqual(['d', 'a', 'c', 'b'], list(gradient_descent.in_random_order(['a', 'b', 'c', 'd'])))
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import unittest
import simple_linear_regression


class TestLinearRegression(unittest.TestCase):
    def test_predict(self):
        self.assertEqual(11, simple_linear_regression.predict(2, 3, 4))

    def test_error(self):
        self.assertEqual(-0.5, simple_linear_regression.error(2, 3, 4, 10.5))

    def test_sum_of_squared_errors(self):
        self.assertEqual(1.25, simple_linear_regression.sum_of_squared_errors(2, 3, [2, 3, 4], [8, 9, 10.5]))

    def test_total_sum_of_squares(self):
        self.assertEqual(16+4+36, simple_linear_regression.total_sum_of_squares([2, 4, 12]))

    def test_r_squared(self):
        self.assertEqual(1.0 - (51 / 56), simple_linear_regression.r_squared(2, 3, [2, 3, 4], [2, 4, 12]))


if __name__ == '__main__':
    unittest.main()

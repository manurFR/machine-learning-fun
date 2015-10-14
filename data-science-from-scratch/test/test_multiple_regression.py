#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import random
import unittest
import mock
import multiple_regression


class TestMultipleRegression(unittest.TestCase):
    def test_predict(self):
        print 3 + 49/5 + 2
        self.assertEqual(3 + 49/5 + 2, multiple_regression.predict([1, 49, 4, 0], [3, 0.2, 0.5, 2]))

    def test_error(self):
        self.assertEqual(10 - 49/5, multiple_regression.error([1, 49, 4, 0], 15, [3, 0.2, 0.5, 2]))

    def test_bootstrap_sample(self):
        with mock.patch.object(random, 'choice', side_effect=[4, 3, 3, 2, 4, 0]):
            self.assertEqual([4, 3, 3, 2, 4], multiple_regression.bootstrap_sample([2, 3, 4, 3, 2]))

    def test_bootstrap_statistic(self):
        with mock.patch.object(random, 'choice', side_effect=[4, 3, 3, 2, 4, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]):
            self.assertEqual([16, 10, 15], multiple_regression.bootstrap_statistic([2, 3, 4, 3, 2], sum, 3))

    def test_ridge_penalty(self):
        self.assertEqual(3 * (2 * 2 + 3 * 3 + 4 * 4), multiple_regression.ridge_penalty([1, 2, 3, 4], 3))


if __name__ == '__main__':
    unittest.main()

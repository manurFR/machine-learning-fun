#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import math
import working_with_data


class MyTestCase(unittest.TestCase):
    def test_bucketize(self):
        self.assertEqual(4, working_with_data.bucketize(5, 2))
        self.assertEqual(21, working_with_data.bucketize(22, 7))

    def test_make_histogram(self):
        self.assertEqual({2: 1, 4: 2, 8: 2}, working_with_data.make_histogram([2, 5, 5, 8, 9], 2))
        self.assertEqual({6: 1}, working_with_data.make_histogram([7], 2))
        self.assertEqual({}, working_with_data.make_histogram([], 2))

    def test_columns_correlation(self):
        self.assertEqual(8 / math.sqrt(2) / math.sqrt(32),
                         working_with_data.columns_correlation([[3, 3, 2], [5, 4, 10]], 0, 2))

    def test_correlation_matrix(self):
        correlation = 8 / math.sqrt(2) / math.sqrt(32)
        # à vérifier
        self.assertEqual([[correlation, correlation], [correlation, correlation]],
                         working_with_data.correlation_matrix([[3, 2], [5, 10]]))


if __name__ == '__main__':
    unittest.main()

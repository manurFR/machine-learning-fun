#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
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
        # correlation([3, 5, 1], [2, 10, 6]) = 0.5
        self.assertEqual([[1, 0.5], [0.5, 1]],
                         working_with_data.correlation_matrix([[3, 2], [5, 10], [1, 6]]))

    def test_parse_row(self):
        self.assertEqual(["1", 2, " 9 "], working_with_data.parse_row(["1", "2", 9], [None, int, lambda x: str(x).center(3)]))
        self.assertEqual([], working_with_data.parse_row([], []))
        self.assertEqual([2.5, None], working_with_data.parse_row(["2.5", "a"], [float, float]))

    def test_parse_rows_with(self):
        self.assertEqual([[1, 6], [3, 12]],
            [row for row in working_with_data.parse_rows_with([["1", "2"], ["3", "4"]], [int, lambda x: int(x) * 3])])

    def test_try_or_none(self):
        self.assertEqual(3.33, working_with_data.try_or_none(float)("3.33"))
        self.assertEqual(None, working_with_data.try_or_none(float)("abc"))

    def test_picker(self):
        self.assertEqual(175, working_with_data.picker("height")({"age": 26, "height": 175, "weight": 65}))

    def test_pluck(self):
        self.assertEqual([15, 53, 26], working_with_data.pluck(
            "age", [{"age": 15, "height": 158}, {"age": 53, "height": 169}, {"age": 26, "height": 175}]))

    def test_group_by(self):
        p1, p2, p3 = {"age": 31, "height": 158}, {"age": 53, "height": 169}, {"age": 31, "height": 175}
        self.assertEqual({31: [p1, p3], 53: [p2]},
                         working_with_data.group_by(lambda r: r["age"], [p1, p2, p3]))

        self.assertEqual({31: 175, 53: 169}, working_with_data.group_by(lambda r: r["age"], [p1, p2, p3],
                                                                        lambda rows: max(r["height"] for r in rows)))

    def test_scale(self):
        self.assertEqual(([1.5, 2], [math.sqrt(0.5), math.sqrt(18)]), working_with_data.scale([[1, 5], [2, -1]]))

    def test_rescale(self):
        self.assertEqual([[-0.5/math.sqrt(0.5), 3/math.sqrt(18)], [0.5/math.sqrt(0.5), -3/math.sqrt(18)]],
                         working_with_data.rescale([[1, 5], [2, -1]]))
        self.assertEqual([[3]], working_with_data.rescale([[3]]))

    def test_de_mean_matrix(self):
        self.assertEqual([[-0.5, 3], [0.5, -3]], working_with_data.de_mean_matrix([[1, 5], [2, -1]]))

    def test_direction(self):
        self.assertEqual([1/3, 2/3, -2/3], working_with_data.direction([1, 2, -2]))

    def test_directional_variance_row(self):
        self.assertEqual(25, working_with_data.directional_variance_row([3, 6, 15], [1, 2, -2]))

    def test_directional_variance(self):
        self.assertEqual(34, working_with_data.directional_variance([[3, 6, 15], [3, 3, 0]], [1, 2, -2]))

if __name__ == '__main__':
    unittest.main()

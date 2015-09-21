#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import random

import unittest
import mock
import machine_learning


class TestMachineLearning(unittest.TestCase):
    def test_split_data(self):
        with mock.patch.object(random, 'random', side_effect=[r/10 for r in range(7)]):
            self.assertEqual(([0, 1, 2, 3], [4, 5, 6]), machine_learning.split_data(range(7), 0.4))

    def test_train_test_split(self):
        with mock.patch.object(random, 'random', side_effect=[r/3 for r in range(3)]):
            self.assertEqual((([1, 2],), ([3, 4], [5, 6]), (10,), (20, 30)),
                             machine_learning.train_test_split([[1, 2], [3, 4], [5, 6]], [10, 20, 30], 0.67))
            
    def test_accuracy(self):
        self.assertEqual(0.98114, machine_learning.accuracy(70, 4930, 13930, 981070))

    def test_precision(self):
        self.assertEqual(0.014, machine_learning.precision(70, 4930, 13930, 981070))

    def test_recall(self):
        self.assertEqual(0.005, machine_learning.recall(70, 4930, 13930, 981070))

    def test_f1_score(self):
        # p = 8/10 = 4/5   r = 8/12 = 2/3   f1 = (16/15) / (22/15) = 16/22
        self.assertAlmostEquals(16/22, machine_learning.f1_score(8, 2, 4, 9999))


if __name__ == '__main__':
    unittest.main()

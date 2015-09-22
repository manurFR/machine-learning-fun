#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import unittest
import nearest_neighbors


class TestKNearestNeighbors(unittest.TestCase):
    def test_majority_vote(self):
        self.assertEqual("jim", nearest_neighbors.majority_vote(["joe", "jim", "dan", "jim", "sam"]))
        self.assertEqual("joe", nearest_neighbors.majority_vote(["joe", "jim", "dan", "joe", "jim"]))

    def test_knn_classify(self):
        self.assertEqual("jim", nearest_neighbors.knn_classify(3, labeled_points=[([1], "joe"), ([5], "joe"),
                                                                                  ([10], "jim"), ([13], "jim"),
                                                                                  ([18], "sam"), ([22], "joe")],
                                                               new_point=[12]))


if __name__ == '__main__':
    unittest.main()

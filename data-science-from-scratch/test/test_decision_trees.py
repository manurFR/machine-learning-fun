#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import unittest
import math

import decision_trees


input1 = ({'key1': 'a', 'key2': 'apple'}, True)
input2 = ({'key1': 'b', 'key2': 'pear'}, True)
input3 = ({'key1': 'c', 'key2': 'apple'}, False)

class TestDecisionTrees(unittest.TestCase):
    def test_entropy(self):
        self.assertEqual(0.0, decision_trees.entropy([0.0, 1.0, 1.0, 0.0]))
        self.assertEqual(- 0.1 * math.log(0.1, 2) - 0.5 * math.log(0.5, 2), decision_trees.entropy([0.0, 1.0, 0.1, 0.5]))

    def test_class_probabilities(self):
        self.assertEqual({1/6, 2/6, 3/6}, set(decision_trees.class_probabilities(['a', 'b', 'b', 'c', 'c', 'c'])))

    def test_data_entropy(self):
        self.assertAlmostEquals(-1/6 * math.log(1/6, 2) - 2/6 * math.log(2/6, 2) - 3/6 * math.log(3/6, 2),
                                decision_trees.data_entropy([('input1', 'a'), ('input2', 'b'), ('input3', 'b'),
                                                             ('input4', 'c'), ('input5', 'c'), ('input6', 'c')]))

    def test_partition_entropy(self):
        self.assertEqual(- math.log(0.5, 2) / 2, decision_trees.partition_entropy([[('i1', 'a'), ('i2', 'b')],
                                                                                   [('i3', 'c'), ('i4', 'c')]]))

    def test_partition_by(self):
        self.assertEqual({'apple': [input1, input3], 'pear': [input2]},
                         decision_trees.partition_by([input1, input2, input3], 'key2'))

    def test_partition_entropy_by(self):
        self.assertEqual(- math.log(0.5, 2) * 2/3, decision_trees.partition_entropy_by([input1, input2, input3], 'key2'))

    def test_classify(self):
        self.assertEqual(True, decision_trees.classify(True, {'input': 'test'}))
        self.assertEqual(False, decision_trees.classify(False, {'input': 'test'}))
        self.assertEqual(True, decision_trees.classify(('att', {'yes': True, 'no': False}), {'att': 'yes'}))
        self.assertEqual(False, decision_trees.classify(('att', {'yes': True, None: False}), {'att': 'unknown'}))

    def test_build_tree_id3_only_one_label(self):
        self.assertEqual(False, decision_trees.build_tree_id3([({'test': 'stuff'}, False), ({'test': 'ba'}, False)]))
        self.assertEqual(True, decision_trees.build_tree_id3([({'test': 'stuff'}, True), ({'test': 'ba'}, True)]))

    def test_build_tree_id3_no_split_candidates_left(self):
        self.assertEqual(True, decision_trees.build_tree_id3(
            [({'test': 'stuff'}, True), ({'test': 'stuff'}, False), ({'test': 'stuff'}, True)], split_candidates=[]))
        self.assertEqual(False, decision_trees.build_tree_id3(
            [({'test': 'stuff'}, True), ({'test': 'stuff'}, False), ({'test': 'stuff'}, False)], split_candidates=[]))
        self.assertEqual(True, decision_trees.build_tree_id3(
            [({'test': 'stuff'}, True), ({'test': 'stuff'}, False)], split_candidates=[]))

    def test_build_tree_id3_real_tree(self):
        self.assertEqual(('difficulty', {'easy': True,
                                         'hard': False,
                                         'medium': ('level', {'1': True, '2': False, None: False}),
                                         None: False}),
                         decision_trees.build_tree_id3([
                             ({'level': '1', 'difficulty': 'easy'}, True),
                             ({'level': '1', 'difficulty': 'medium'}, True),
                             ({'level': '1', 'difficulty': 'hard'}, False),
                             ({'level': '2', 'difficulty': 'easy'}, True),
                             ({'level': '2', 'difficulty': 'medium'}, False),
                             ({'level': '2', 'difficulty': 'hard'}, False)
                         ]))


if __name__ == '__main__':
    unittest.main()

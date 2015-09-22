#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import unittest
import math
import naive_bayes


class TestNaiveBayes(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual({'hello', 'it\'s', 'time', 'for', '1000s', 'of'},
                         naive_bayes.tokenize("Hello it's time for 1000s of HELLO"))

    def test_count_words(self):
        self.assertEqual({'hello': [1, 1], 'world': [0, 2], 'cheap': [1, 0], 'viagra': [2, 0], 'big': [0, 1]},
                         naive_bayes.count_words([("hello world", False), ("Cheap Viagra", True),
                                                  ("viagra hello", True), ("big big WORLD", False)]))

    def test_word_probabilities(self):
        self.assertEqual({('hello', 1.5/3, 1.5/4), ('world', 0.5/3, 2.5/4), ('viagra', 2.5/3, 0.5/4)},
                         set(naive_bayes.word_probabilities({'hello': [1, 1], 'world': [0, 2], 'viagra': [2, 0]},
                                                            total_spams=2, total_non_spams=3, k=0.5)))

    def test_spam_probability(self):
        prob_if_spam = math.exp(math.log(0.1) + math.log(0.8) + math.log(0.9))
        prob_if_not_spam = math.exp(math.log(0.8) + math.log(0.3) + math.log(0.1))
        self.assertAlmostEquals(prob_if_spam / (prob_if_spam + prob_if_not_spam),
                    naive_bayes.spam_probability([('hello', 0.1, 0.8), ('world', 0.2, 0.7), ('viagra', 0.9, 0.1)],
                                                   "Hello, this is Joe, your Viagra seller"))


if __name__ == '__main__':
    unittest.main()

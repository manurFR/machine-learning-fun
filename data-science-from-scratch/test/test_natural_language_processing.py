#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import random
import unittest
import mock
import natural_language_processing

class TestNaturalLanguageProcessing(unittest.TestCase):
    def test_fix_unicode(self):
        self.assertEqual(u"It's so cool", natural_language_processing.fix_unicode(u"It\u2019s so cool"))

    def test_generate_gibberish_using_bigrams(self):
        with mock.patch.object(random, 'choice', side_effect=["I'm", "glad", "you", "will", "start", "."]):
            self.assertEqual("I'm glad you will start.", natural_language_processing.generate_gibberish_using_bigrams({
                ".": ["Hello", "I'm", "Today"],
                "start": [".", "the"], "you": ["are", "said", "will"],
                "I'm": ["glad"], "glad": ["you", "part"], "will": ["start"]
            }))

    def test_is_terminal(self):
        self.assertFalse(natural_language_processing.is_terminal("_N"))
        self.assertFalse(natural_language_processing.is_terminal("_NP _VP"))
        self.assertTrue(natural_language_processing.is_terminal("hello"))

    def test_expand_simplest(self):
        self.assertEqual([], natural_language_processing.expand({"_S": ["hello"]}, []))
        self.assertEqual(["hello"], natural_language_processing.expand({"_S": ["hello"]}, ["_S"]))

    def test_expand_full(self):
        with mock.patch.object(random, 'choice', side_effect=["_N _VP", "Ben", "_V _W", "eats", "cheese"]):
            self.assertEqual(["Ben", "eats", "cheese"], natural_language_processing.expand({
                "_S":  ["_N _VP"],
                "_N":  ["Andy", "Ben"],
                "_VP": ["_V", "_V _W"],
                "_V":  ["eats"],
                "_W":  ["cheese", "jam"]
            }, ["_S"]))

if __name__ == '__main__':
    unittest.main()

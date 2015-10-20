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


if __name__ == '__main__':
    unittest.main()

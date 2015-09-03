from __future__ import division
from unittest import TestCase
import math
import mock
import random
import probability


# noinspection PyUnresolvedReferences
class TestProbability(TestCase):
    def test_uniform_pdf(self):
        self.assertEqual(0, probability.uniform_pdf(-1))
        self.assertEqual(1, probability.uniform_pdf(0))
        self.assertEqual(1, probability.uniform_pdf(0.5))
        self.assertEqual(1, probability.uniform_pdf(0.99))
        self.assertEqual(0, probability.uniform_pdf(1))
        self.assertEqual(0, probability.uniform_pdf(10))

    def test_uniform_cdf(self):
        self.assertEqual(0, probability.uniform_cdf(-1))
        self.assertEqual(0, probability.uniform_cdf(0))
        self.assertEqual(0.5, probability.uniform_cdf(0.5))
        self.assertEqual(1, probability.uniform_cdf(1))
        self.assertEqual(1, probability.uniform_cdf(10))

    def test_normal_pdf(self):
        self.assertEqual(math.exp(0) / math.sqrt(2 * math.pi), probability.normal_pdf(0, mu=0, sigma=1))
        self.assertEqual(math.exp(-1/2) / math.sqrt(2 * math.pi), probability.normal_pdf(1, mu=0, sigma=1))
        self.assertEqual(math.exp(-1/2) / math.sqrt(2 * math.pi), probability.normal_pdf(-1, mu=0, sigma=1))
        self.assertEqual(math.exp(-9/32) / (4 * math.sqrt(2 * math.pi)), probability.normal_pdf(5, mu=2, sigma=4))

    def test_normal_cdf(self):
        self.assertEqual((1 + math.erf(0)) / 2, probability.normal_cdf(0, mu=0, sigma=1))
        self.assertEqual((1 + math.erf(-1 / (3 * math.sqrt(2)))) / 2, probability.normal_cdf(1, mu=2, sigma=3))

    def test_inverse_normal_cdf(self):
        self.assertAlmostEqual(1, probability.inverse_normal_cdf(p=(1 + math.erf(-1 / (3 * math.sqrt(2)))) / 2, mu=2, sigma=3), places=4)

    def test_bernoulli_trial(self):
        with mock.patch.object(random, 'random', return_value=0.5):
            self.assertEqual(0, probability.bernouilli_trial(0))
            self.assertEqual(0, probability.bernouilli_trial(0.5))
            self.assertEqual(1, probability.bernouilli_trial(0.51))
            self.assertEqual(1, probability.bernouilli_trial(1))

    def test_binomial(self):
        with mock.patch.object(random, 'random', return_value=0.5):
            self.assertEqual(0, probability.binomial(10, 0))
            self.assertEqual(10, probability.binomial(10, 0.6))
        self.assertLessEqual(probability.binomial(100, 0.9), 100)
        self.assertEqual(100, probability.binomial(100, 1))
from __future__ import division
from unittest import TestCase
import math
import statistics


class TestStatistics(TestCase):
    def test_mean(self):
        self.assertEqual(2.5, statistics.mean([1, 2, 3, 4]))
        self.assertEqual(3, statistics.mean([3]))
        self.assertEqual(0, statistics.mean([]))
        self.assertEqual(1, statistics.mean([-4, 6]))

    def test_median(self):
        self.assertEqual(4, statistics.median([4, 21, 1, 8, 2]))
        self.assertEqual(4.5, statistics.median([4, 21, 1, 5, 8, 2]))
        self.assertEqual(2, statistics.median([2]))
        self.assertEqual(0, statistics.median([]))

    def test_quantile(self):
        self.assertEqual(3, statistics.quantile([98, 1, 45, 3, 5, 58, 9, 20, 77, 32], 0.10))
        self.assertEqual(5, statistics.quantile([98, 1, 45, 3, 5, 58, 9, 20, 77, 32], 0.25))
        self.assertEqual(58, statistics.quantile([98, 1, 45, 3, 5, 58, 9, 20, 77, 32], 0.75))
        self.assertEqual(99, statistics.quantile([98, 1, 45, 3, 5, 58, 9, 20, 77, 32], 1.0))
        self.assertEqual(1, statistics.quantile([98, 1, 45, 3, 5, 58, 9, 20, 77, 32], 0.01))
        self.assertEqual(1, statistics.quantile([98, 1, 45, 3, 5, 58, 9, 20, 77, 32], 0))

    def test_mode(self):
        self.assertEqual([6], statistics.mode([1, 3, 6, 6, 2, 9, 10, 1, 6, 7, 54, 0]))
        self.assertEqual([1, 6], statistics.mode([3, 6, 1, 6, 2, 9, 10, 1, 6, 7, 54, 1, 0]))
        self.assertEqual([], statistics.mode([]))
        self.assertEqual([1, 2, 3], statistics.mode([3, 2, 1]))

    def test_data_range(self):
        self.assertEqual(12, statistics.data_range([9, 6, 3, 8, 15, 6]))
        self.assertEqual(0, statistics.data_range([6]))
        self.assertEqual(0, statistics.data_range([]))

    def test_de_mean(self):
        self.assertEqual([-3, -2, -1, 0, 0, 6], statistics.de_mean([1, 2, 3, 4, 4, 10]))  # mean is 4
        self.assertEqual([0], statistics.de_mean([1]))
        self.assertEqual([], statistics.de_mean([]))

    def test_variance(self):
        self.assertEqual(((-3*-3)+(-2*-2)+(-1*-1)+(6*6))/5, statistics.variance([1, 2, 3, 4, 4, 10]))
        self.assertEqual(0, statistics.variance([1]))
        self.assertEqual(0, statistics.variance([]))

    def test_standard_deviation(self):
        self.assertEqual(math.sqrt(10), statistics.standard_deviation([1, 2, 3, 4, 4, 10]))
        self.assertEqual(0, statistics.standard_deviation([1]))
        self.assertEqual(0, statistics.standard_deviation([]))

    def test_interquartile_range(self):
        self.assertEqual(53, statistics.interquartile_range([98, 1, 45, 3, 5, 58, 9, 20, 77, 32]))

    def test_covariance(self):
        self.assertEqual(((-3*-3)+(-2*1)+(-1*-6)+(0*-5)+(0*7)+(6*6))/5,
                         statistics.covariance([1, 2, 3, 4, 4, 10], [3, 7, 0, 1, 13, 12]))  # means are 4 and 6
        self.assertEqual(0, statistics.covariance([2], [3]))
        self.assertEqual(0, statistics.covariance([], []))

    def test_correlation(self):
        # covariance: [3, 5] [2, 10] => [-1 1] [-4 4] => 8
        # stddev:    sqrt(2) sqrt(32)
        self.assertEqual(8 / math.sqrt(2) / math.sqrt(32), statistics.correlation([3, 5], [2, 10]))
        self.assertEqual(0, statistics.correlation([3, 3], [2, 10]))

        num_friends = [100,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        daily_minutes = [1,68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]
        self.assertAlmostEqual(0.25, statistics.correlation(num_friends, daily_minutes), places=2)


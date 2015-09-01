from unittest import TestCase
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

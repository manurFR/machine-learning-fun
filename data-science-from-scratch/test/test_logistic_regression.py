from __future__ import division
import unittest
import logistic_regression


class TestLogisticRegression(unittest.TestCase):
    def test_logistic(self):
        self.assertAlmostEqual(0, logistic_regression.logistic(-10), places=4)
        self.assertEqual(0.5, logistic_regression.logistic(0))
        self.assertAlmostEqual(1, logistic_regression.logistic(10), places=4)


if __name__ == '__main__':
    unittest.main()

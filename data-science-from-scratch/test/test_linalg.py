from __future__ import division
from unittest import TestCase

import linalg


class TestVectors(TestCase):
    def test_vector_add(self):
        self.assertEqual([3, 3], linalg.vector_add([1, 2], [2, 1]))
        self.assertEqual([4, 5, 4], linalg.vector_add([-1, 2, 4], [5, 3, 0]))
        self.assertEqual([10], linalg.vector_add([2], [8]))
        self.assertEqual([], linalg.vector_add([], []))

        with self.assertRaises(ValueError):
            linalg.vector_add([1, 2], [3, 4, 8])

    def test_vector_substract(self):
        self.assertEqual([2, 5], linalg.vector_substract([7, 7], [5, 2]))
        self.assertEqual([4, 5], linalg.vector_substract([3, 7], [-1, 2]))
        self.assertEqual([], linalg.vector_substract([], []))

        with self.assertRaises(ValueError):
            linalg.vector_substract([1, 2], [3, 4, 8])

    def test_vector_sum(self):
        self.assertEqual([4, 8, 0], linalg.vector_sum([[1, 2, 3], [3, 2, 2], [0, 4, -5]]))

        with self.assertRaises(ValueError):
            linalg.vector_sum([[1, 2], [3, 4, 8], [5, 6]])

    def test_scalar_multiply(self):
        self.assertEqual([-3, 9, 0], linalg.scalar_multiply(3, [-1, 3, 0]))
        self.assertEqual([], linalg.scalar_multiply(3, []))

    def test_vector_mean(self):
        self.assertEqual([2, 5/4], linalg.vector_mean([[2, 4], [0, -1], [4, 3], [2, -1]]))

    def test_dot_product(self):
        self.assertEqual(8, linalg.dot([1, 2, 0, -1], [4, 4, 4, 4]))

    def test_sum_of_squares(self):
        self.assertEqual(10, linalg.sum_of_squares([3, 1, 0]))

    def test_magnitude(self):
        self.assertEqual(5, linalg.magnitude([4, 3]))
        self.assertEqual(2, linalg.magnitude([2]))
        self.assertEqual(0, linalg.magnitude([]))

    def test_distance(self):
        self.assertEqual(5, linalg.distance([5, 2], [1, -1]))
        self.assertEqual(0, linalg.distance([0, 1], [0, 1]))
        self.assertEqual(2, linalg.distance([5], [3]))
        self.assertEqual(0, linalg.distance([], []))


class TestMatrices(TestCase):
    def test_shape(self):
        self.assertEqual((3, 2), linalg.shape([[1, 1], [2, 2], [3, 3]]))
        self.assertEqual((3, 0), linalg.shape([[], [], []]))
        self.assertEqual((0, 0), linalg.shape([]))

    def test_get_row(self):
        self.assertEqual([3, 4], linalg.get_row([[1, 2], [3, 4], [5, 6]], 1))
        self.assertEqual([], linalg.get_row([[], [], []], 1))

        with self.assertRaises(IndexError):
            linalg.get_row([[1, 2]], 1)

    def test_get_column(self):
        self.assertEqual([2, 4, 6], linalg.get_column([[1, 2], [3, 4], [5, 6]], 1))
        self.assertEqual([1, 1, 3], linalg.get_column([[1], [1], [3]], 0))

        with self.assertRaises(IndexError):
            linalg.get_column([[1, 2]], 3)

    def test_make_matrix(self):
        self.assertEqual([[0, 2, 4], [1, 3, 5]], linalg.make_matrix(2, 3, lambda row, col: row + col*2))
        self.assertEqual([[0, 0], [0, 0]], linalg.make_matrix(2, 2, lambda row, col: 0))
        self.assertEqual([[], [], []], linalg.make_matrix(3, 0, lambda row, col: 8))
        self.assertEqual([], linalg.make_matrix(0, 0, lambda row, col: 5))

    def test_is_diagonal(self):
        self.assertEqual(1, linalg.is_diagonal(0, 0))
        self.assertEqual(1, linalg.is_diagonal(3, 3))
        self.assertEqual(0, linalg.is_diagonal(1, 2))
        self.assertEqual(0, linalg.is_diagonal(4, -4))
from __future__ import division
import network_analysis
import unittest


class TestNetworkAnalysis(unittest.TestCase):
    def test_initialize_frontier(self):
        user = {"id": "mike", "friends": ["john", "amy", "don"]}
        frontier = network_analysis.initialize_frontier(user)
        self.assertEqual((user, "john"), frontier.popleft())
        self.assertEqual((user, "amy"), frontier.popleft())
        self.assertEqual((user, "don"), frontier.popleft())

    def test_shortest_paths_from(self):
        #    0 -- 1
        #    \    \ `.
        #    2 -- 3 --`4
        users = [{"id": id} for id in [0, 1, 2, 3, 4]]
        users[0]["friends"] = [users[1], users[2]]
        users[1]["friends"] = [users[0], users[3], users[4]]
        users[2]["friends"] = [users[0], users[3]]
        users[3]["friends"] = [users[1], users[2], users[4]]
        users[4]["friends"] = [users[1], users[3]]

        paths = network_analysis.shortest_paths_from(users[3])
        self.assertEqual([[]], paths[3])
        self.assertEqual([[1, 0], [2, 0]], paths[0])
        self.assertEqual([[1]], paths[1])
        self.assertEqual([[2]], paths[2])
        self.assertEqual([[4]], paths[4])

    def test_min_path_length(self):
        self.assertEqual(3, network_analysis.min_path_length([[1, 2, 3], [5, 6, 7]]))
        self.assertEqual(float('inf'), network_analysis.min_path_length(None))

    def test_farness(self):
        self.assertEqual(5, network_analysis.farness({"shortest_paths": {0: [[]], 1: [[3, 2, 1]], 2: [[3, 2], [4, 2]]}}))

    def test_matrix_product_entry(self):
        self.assertEqual(3*-1+4*2, network_analysis.matrix_product_entry([[1, 2], [3, 4]], [[4, -1, 3], [2, 2, -2]], 1, 1))

    def test_matrix_product(self):
        self.assertEqual([[8, 3, -1], [20, 5, 1]],
                         network_analysis.matrix_product([[1, 2], [3, 4]], [[4, -1, 3], [2, 2, -2]]))

    def test_vector_as_matrix(self):
        self.assertEqual([[1], [2], [3]], network_analysis.vector_as_matrix([1, 2, 3]))

    def test_vector_from_matrix(self):
        self.assertEqual([1, 2, 3], network_analysis.vector_from_matrix([[1], [2], [3]]))

    def test_matrix_operate(self):
        self.assertEqual([1, 7], network_analysis.matrix_operate([[1, 2, -1], [3, 4, -2]], [5, 1, 6]))

    def test_page_rank(self):
        users = [{"id": 1, "endorses": [{"id": 2}]},
                 {"id": 2, "endorses": [{"id": 1}, {"id": 4}]},
                 {"id": 3, "endorses": [{"id": 1}]},
                 {"id": 4, "endorses": []}]
        self.assertEqual({1: 1.425/4, 2: 1/4, 3: 0.15/4, 4: 0.575/4},
                         {key: round(value, 5) for key, value in
                          network_analysis.page_rank(users, damping=0.85, num_iters=1).iteritems()})

if __name__ == '__main__':
    unittest.main()

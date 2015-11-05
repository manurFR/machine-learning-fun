import network_analysis

__author__ = 'ebizieau'

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

if __name__ == '__main__':
    unittest.main()

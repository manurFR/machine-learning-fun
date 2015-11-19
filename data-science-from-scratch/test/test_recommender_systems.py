from __future__ import division
from unittest import TestCase
import recommender_systems


class TestRecommenderSystems(TestCase):
    def test_make_user_interest_vector(self):
        self.assertEqual([1, 0, 0, 1], recommender_systems.make_user_interest_vector(interests=['a', 'b', 'c', 'd'],
                                                                                     user_interests=['a', 'd']))

    def test_most_similar_users_to(self):
        self.assertEqual([(2, 0.9), (4, 0.6), (1, 0.3)],
                         recommender_systems.most_similar_users_to(user_similarities=[[1.0, 0.3, 0.9, 0.0, 0.6]],
                                                                   user_id=0))
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from collections import defaultdict
from functools import partial
import math
from linalg import dot


def cosine_similarity(v, w):
    return dot(v, w) / math.sqrt(dot(v, v) * dot(w, w))


def make_user_interest_vector(interests, user_interests):
    return [1 if interest in user_interests else 0 for interest in interests]


def most_similar_users_to(user_similarities, user_id):
    pairs = [(other_user_id, similarity) for other_user_id, similarity in enumerate(user_similarities[user_id])
             if user_id != other_user_id and similarity > 0]
    return sorted(pairs, key=lambda (_, similarity): similarity, reverse=True)


def user_based_suggestions(user_similarities, users_interests, user_id, include_current_interests=False):
    suggestions = defaultdict(float)
    for other_user_id, similarity in most_similar_users_to(user_similarities, user_id):
        for interest in users_interests[other_user_id]:
            suggestions[interest] += similarity

    suggestions = sorted(suggestions.items(), key=lambda (_, weight): weight, reverse=True)

    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight) for suggestion, weight in suggestions if suggestion not in users_interests[user_id]]


if __name__ == '__main__':
    users_interests = [
        ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
        ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
        ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
        ["R", "Python", "statistics", "regression", "probability"],
        ["machine learning", "regression", "decision trees", "libsvm"],
        ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
        ["statistics", "probability", "mathematics", "theory"],
        ["machine learning", "scikit-learn", "Mahout", "neural networks"],
        ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
        ["Hadoop", "Java", "MapReduce", "Big Data"],
        ["statistics", "R", "statsmodels"],
        ["C++", "deep learning", "artificial intelligence", "probability"],
        ["pandas", "R", "Python"],
        ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
        ["libsvm", "regression", "support vector machines"]
    ]

    unique_interests = sorted(list({interest for user_interests in users_interests for interest in user_interests}))

    user_interest_matrix = map(partial(make_user_interest_vector, unique_interests), users_interests)

    user_similarities = [[cosine_similarity(interest_vector_i, interest_vector_j) for interest_vector_j
                          in user_interest_matrix] for interest_vector_i in user_interest_matrix]

    print most_similar_users_to(user_similarities, 0)

    print user_based_suggestions(user_similarities, users_interests, 0)
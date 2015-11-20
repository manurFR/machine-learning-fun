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


def most_similar_interests_to(interest_similarities, interest_id, unique_interests):
    pairs = [(unique_interests[other_interest_id], similarity)
             for other_interest_id, similarity in enumerate(interest_similarities[interest_id])
             if interest_id != other_interest_id and similarity > 0]
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


def item_based_suggestions(interest_similarities, users_interests, user_interest_matrix, unique_interests, user_id, include_current_interests=False):
    suggestions = defaultdict(float)
    for interest_id, is_interested in enumerate(user_interest_matrix[user_id]):
        if is_interested == 1:
            for interest, similarity in most_similar_interests_to(interest_similarities, interest_id, unique_interests):
                suggestions[interest] += similarity

    suggestions = sorted(suggestions.items(), key=lambda (_, similarity): similarity, reverse=True)

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

    # item-based
    print
    interest_user_matrix = [[user_interest_vector[j] for user_interest_vector in user_interest_matrix]
                            for j, _ in enumerate(unique_interests)]
    interest_similarities = [[cosine_similarity(user_vector_i, user_vector_j) for user_vector_j
                              in interest_user_matrix] for user_vector_i in interest_user_matrix]

    print most_similar_interests_to(interest_similarities, 0, unique_interests)

    print item_based_suggestions(interest_similarities, users_interests, user_interest_matrix, unique_interests, 0)

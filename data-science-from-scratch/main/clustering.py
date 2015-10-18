#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import random
from linalg import squared_distance, vector_mean
import matplotlib.pyplot as plt

class KMeans(object):
    def __init__(self, k):
        self.k = k
        self.means = None

    def classify(self, input):
        """return the index of the cluster closest to the input"""
        return min(range(self.k), key=lambda i: squared_distance(input, self.means[i]))

    def train(self, inputs):
        self.means = random.sample(inputs, self.k)
        assignments = None

        while True:
            # assign a cluster (through its mean) to each input
            new_assignments = map(self.classify, inputs)

            # if no assignments have changed, we have converged
            if assignments == new_assignments:
                return

            # otherwise keep the new assignments
            assignments = new_assignments

            # and compute the new means
            for i in range(self.k):
                # gather the points whose cluster is the current one
                points = [point for point, mean in zip(inputs, assignments) if mean == i]

                # make sure each cluster is not empty before computing the new mean
                if points:
                    self.means[i] = vector_mean(points)


def squared_clustering_errors(inputs, k):
    """returns the total squared error for clustering the inputs in k clusters via k-means"""
    clusterer = KMeans(k)
    clusterer.train(inputs)
    assignments = map(clusterer.classify, inputs)

    return sum(squared_distance(input, clusterer.means[cluster]) for input, cluster in zip(inputs, assignments))

if __name__ == '__main__':
    def rounded(value):
        try:
            return [round(element) for element in value]
        except TypeError:
            return round(value)

    locations = [[-14, -5], [13, 13], [20, 23], [-19, -11], [-9, -16], [21, 27], [-49, 15], [26, 13], [-46, 5],
                 [-34, -1], [11, 15], [-49, 0], [-22, -16], [19, 28], [-12, -8], [-13, -19], [-41, 8], [-11, -6],
                 [-25, -9], [-18, -3]]

    random.seed(0)
    clusterer = KMeans(3)
    clusterer.train(locations)
    print "3 meetups :", [rounded(mean) for mean in clusterer.means]

    random.seed(0)
    clusterer = KMeans(2)
    clusterer.train(locations)
    print "2 meetups :", [rounded(mean) for mean in clusterer.means]

    # random.seed(0)
    ks = range(1, len(locations) + 1)
    errors = [squared_clustering_errors(locations, k) for k in ks]

    plt.plot(ks, errors)
    plt.xticks(ks)
    plt.xlabel("k")
    plt.ylabel("total squared error")
    plt.title("Total Error vs. # of Clusters")
    plt.show()
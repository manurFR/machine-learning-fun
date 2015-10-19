#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import pprint
import random
from linalg import squared_distance, vector_mean, distance
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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


def is_leaf(cluster):
    return len(cluster) == 1


def get_children(cluster):
    if is_leaf(cluster):
        raise TypeError("a leaf cluster has no children")
    else:
        return cluster[1]


def get_values(cluster):
    if is_leaf(cluster):
        return cluster
    else:
        return [value for child in get_children(cluster) for value in get_values(child)]


def cluster_distance(cluster1, cluster2, distance_func=min):
    return distance_func([distance(input1, input2) for input1 in get_values(cluster1) for input2 in get_values(cluster2)])


def get_merge_order(cluster):
    if is_leaf(cluster):
        return float('inf')
    else:
        return cluster[0]


def bottom_up_cluster(inputs, distance_func=min):
    clusters = [(input,) for input in inputs]  # initially, each input is a (leaf) cluster

    # as long as we have more than one cluster left
    while len(clusters) > 1:
        closest1, closest2 = min([(cluster1, cluster2) for i, cluster1 in enumerate(clusters) for cluster2 in clusters[:i]],
                                 key=lambda (x, y): cluster_distance(x, y, distance_func))

        # remove the two closest clusters from the list...
        clusters = [c for c in clusters if c != closest1 and c != closest2]

        # ...merge them...
        merged_cluster = (len(clusters), [closest1, closest2])

        # ...and add the merge
        clusters.append(merged_cluster)

    # return the final state
    return clusters[0]


def generate_clusters(base_cluster, num_clusters):
    clusters = [base_cluster]

    while len(clusters) < num_clusters:
        next_cluster = min(clusters, key=get_merge_order)
        clusters = [cluster for cluster in clusters if cluster != next_cluster]
        clusters.extend(get_children(next_cluster))

    return clusters



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

    # clustering colors
    if True:
        fileimage = r'godhelpthegirl.jpg'
        img = mpimg.imread(fileimage)
        red, green, blue = img[0][0]  # top left pixel
        print red, green, blue

        pixels = [pixel for row in img for pixel in row]
        print len(pixels), "pixels to analyse"
        clusterer = KMeans(5)
        clusterer.train(pixels)

        def recolor(pixl):
            index = clusterer.classify(pixl)
            return clusterer.means[index]

        new_img = [[recolor(pixel) for pixel in row] for row in img]
        plt.imshow(new_img)
        plt.axis('off')
        plt.show()

    # hierarchical clustering
    base_cluster = bottom_up_cluster(locations, min)
    pprint.pprint(base_cluster)

    three_clusters = [get_values(cluster) for cluster in generate_clusters(base_cluster, 3)]

    for i, cluster, marker, color in zip([1, 2, 3],
                                         three_clusters,
                                         ['D', 'o', '*'],
                                         ['r', 'g', 'b']):
        xs, ys = zip(*cluster)
        plt.scatter(xs, ys, color=color, marker=marker)

        x, y = vector_mean(cluster)
        plt.plot(x, y, marker='$' + str(i) + '$', color='black')

    plt.title("3 clusters")
    plt.show()

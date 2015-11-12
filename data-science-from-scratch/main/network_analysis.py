from __future__ import division
from collections import deque
from functools import partial
import pprint
import random
from linalg import dot, get_row, get_column, shape, make_matrix, magnitude, scalar_multiply, distance


def initialize_frontier(from_user):
    return deque((from_user, friend) for friend in from_user["friends"])


def min_path_length(paths):
    # what's the shortest path to here, so far ?
    if paths:
        min_length = len(paths[0])
    else:
        min_length = float('inf')

    return min_length


def explore_user(curr_user, prev_user, shortest_paths_to):
    curr_user_id = curr_user["id"]

    # because of the way we're adding to the queue, necessarily we already know some shortest paths to prev_user
    new_paths_to_curr_user = [path + [curr_user_id] for path in shortest_paths_to[prev_user["id"]]]
    old_paths_to_curr_user = shortest_paths_to.get(curr_user_id, [])  # we may already know a shortest path

    # only keep paths that aren't too long and are actually new
    new_paths_to_curr_user = [path for path in new_paths_to_curr_user
                              if len(path) <= min_path_length(old_paths_to_curr_user)
                              and path not in old_paths_to_curr_user]

    shortest_paths_to[curr_user_id] = old_paths_to_curr_user + new_paths_to_curr_user


def shortest_paths_from(from_user):
    """returns the shortest paths from from_user to all other users as a dict whose keys are the other users' ids,
         and their value a list of each of the shortest paths (one or more)"""

    # prepare the dict with the path of from_user to herself, which is naturally void
    shortest_paths_to = {from_user["id"]: [[]]}

    # other users that we still need to check, initialized with all "first level" friends of from_user
    frontier = initialize_frontier(from_user)

    while frontier:
        prev_user, curr_user = frontier.popleft()

        explore_user(curr_user, prev_user, shortest_paths_to)

        # add new neighbours to the frontier
        frontier.extend((curr_user, friend) for friend in curr_user["friends"] if friend["id"] not in shortest_paths_to)

    return shortest_paths_to


def farness(user):
    return sum(len(paths[0]) for paths in user["shortest_paths"].values())


def matrix_product_entry(A, B, i, j):
    return dot(get_row(A, i), get_column(B, j))


def matrix_product(A, B):
    rows_A, cols_A = shape(A)
    rows_B, cols_B = shape(B)
    if cols_A != rows_B:
        raise ArithmeticError("incompatible shapes")

    return make_matrix(rows_A, cols_B, partial(matrix_product_entry, A, B))


def vector_as_matrix(v):
    """transform the vector (represented as a liste) to a n x 1 matrix"""
    return [[elt] for elt in v]


def vector_from_matrix(v_as_matrix):
    return [row[0] for row in v_as_matrix]


def matrix_operate(A, v):
    """given A of shape n x k and v vector of length k, transform v to a vector of length n"""
    v_as_matrix = vector_as_matrix(v)
    product = matrix_product(A, v_as_matrix)
    return vector_from_matrix(product)


def find_eigenvector(A, tolerance=0.00001):
    guess = [random.random() for _ in A]

    while True:
        result = matrix_operate(A, guess)
        length = magnitude(result)
        next_guess = scalar_multiply(1/length, result)  # rescale to a magnitude of 1

        if distance(guess, next_guess) < tolerance:
            return next_guess, length  # eigenvector, eigenvalue
        guess = next_guess


def page_rank(users, damping=0.85, num_iters=100):
    pr = {user["id"]: 1/len(users) for user in users}

    # fraction of PageRank that is redistributed
    base_pr = (1 - damping) / len(users)

    for _ in range(num_iters):
        next_pr = {user["id"]: base_pr for user in users}
        for user in users:
            links_pr = pr[user["id"]] * damping
            for endorsee in user["endorses"]:
                next_pr[endorsee["id"]] += links_pr / len(user["endorses"])

        pr = next_pr

    return pr


if __name__ == '__main__':
    users = [
        {"id": 0, "name": "Hero"},
        {"id": 1, "name": "Dunn"},
        {"id": 2, "name": "Sue"},
        {"id": 3, "name": "Chi"},
        {"id": 4, "name": "Thor"},
        {"id": 5, "name": "Clive"},
        {"id": 6, "name": "Hicks"},
        {"id": 7, "name": "Devin"},
        {"id": 8, "name": "Kate"},
        {"id": 9, "name": "Klein"}
    ]

    friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
                   (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

    # give each user a friends list
    for user in users:
        user["friends"] = []

    # and populate it
    for i, j in friendships:
        # this works because users[i] is the user whose id is i
        users[i]["friends"].append(users[j]) # add i as a friend of j
        users[j]["friends"].append(users[i]) # add j as a friend of i

    for user in users:
        user["shortest_paths"] = shortest_paths_from(user)
        user["betweenness_centrality"] = 0.0
        user["closeness_centrality"] = 1 / farness(user)

    for source in users:
        source_id = source["id"]
        for target_id, paths in source["shortest_paths"].iteritems():
            if source_id < target_id:
                num_paths = len(paths)
                contrib = 1 / num_paths
                for path in paths:
                    for id in path:
                        if id not in [source_id, target_id]:
                            users[id]["betweenness_centrality"] += contrib

    print "Betweenness centrality"
    for user in users:
        print user["id"], ' : ', user["betweenness_centrality"]
    print

    print "Closeness centrality"
    for user in users:
        print user["id"], ' : ', user["closeness_centrality"]
    print

    # eigenvector centrality
    def adjacency_fn(i, j):
        return 1 if (i, j) in friendships or (j, i) in friendships else 0
    adjacency_matrix = make_matrix(len(users), len(users), adjacency_fn)

    eigenvector_centralities, _ = find_eigenvector(adjacency_matrix)

    pprint.pprint(adjacency_matrix)
    print eigenvector_centralities

    # directed graphs
    print

    endorsements = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1), (1, 3),
                    (2, 3), (3, 4), (5, 4), (5, 6), (7, 5), (6, 8), (8, 7), (8, 9)]

    for user in users:
        user["endorses"] = []       # add one list to track outgoing endorsements
        user["endorsed_by"] = []    # and another to track endorsements

    for source_id, target_id in endorsements:
        users[source_id]["endorses"].append(users[target_id])
        users[target_id]["endorsed_by"].append(users[source_id])

    endorsements_by_id = [(user["id"], len(user["endorsed_by"]))
                          for user in users]

    sorted(endorsements_by_id,
           key=lambda (user_id, num_endorsements): num_endorsements,
           reverse=True)

    print "endorsements : ", endorsements_by_id

    print "PageRank"
    for user_id, pr in page_rank(users).iteritems():
        print user_id, pr

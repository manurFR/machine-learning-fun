#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict, Counter
import random
import re
from bs4 import BeautifulSoup
import requests
import sys


def fix_unicode(text):
    return text.replace(u"\u2019", "'")


def generate_gibberish_using_bigrams(bigrams):
    current = "."  # start with a word that comes after a dot
    result = []
    while True:
        next_word_candidates = bigrams[current]
        current = random.choice(next_word_candidates)
        result.append(current)
        if current == ".":
            return " ".join(result).replace(" .", ".")


def generate_gibberish_using_trigrams(trigrams, starts):
    current = random.choice(starts)
    previous = "."
    result = [current]
    while True:
        next_word_candidates = trigrams[(previous, current)]
        next_word = random.choice(next_word_candidates)

        previous, current = current, next_word
        result.append(current)

        if current == ".":
            return " ".join(result).replace(" .", ".")


def is_terminal(token):
    return token[0] != '_'


def expand(grammar, tokens):
    for i, token in enumerate(tokens):
        if is_terminal(token):
            continue

        replacement = random.choice(grammar[token])

        if is_terminal(replacement):
            tokens[i] = replacement
        else:
            tokens = tokens[:i] + replacement.split() + tokens[(i+1):]

        return expand(grammar, tokens)
    return tokens


def sample_from(weights):
    """returns i with probability weights[i] / sum(weights)"""
    total = sum(weights)
    rnd = total * random.random()
    for i, w in enumerate(weights):
        rnd -= w
        if rnd <= 0:
            return i


def p_topic_given_document(K, document_topic_counts, document_lengths, topic, doc, alpha=0.1):
    """the fraction of words in document doc that are assigned to topic, plus some smoothing"""
    return (document_topic_counts[doc][topic] + alpha) / (document_lengths[doc] + K * alpha)


def p_word_given_topic(W, topic_word_counts, topic_counts, word, topic, beta=0.1):
    """the fraction of words assigned to topic that equal word, plus some smoothing"""
    return (topic_word_counts[topic][word] + beta) / (topic_counts[topic] + W * beta)


def topic_weight(K, document_topic_counts, document_lengths, W, topic_word_counts, topic_counts, d, word, k):
    """given a document and a word in that document, return the weight for the kth topic"""
    return p_word_given_topic(W, topic_word_counts, topic_counts, word, k) * \
           p_topic_given_document(K, document_topic_counts, document_lengths, k, d)


def choose_new_topic(K, document_topic_counts, document_lengths, W, topic_word_counts, topic_counts, d, word):
    return sample_from([topic_weight(K, document_topic_counts, document_lengths, W, topic_word_counts, topic_counts, d, word, k) for k in range(K)])



if __name__ == '__main__':
    url = "http://radar.oreilly.com/2010/06/what-is-data-science.html"
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html5lib')
    content = soup.find("div", "article-body")

    regex = r"[\w']+|[\.]"
    document = []
    for paragraph in content("p"):
        words = re.findall(regex, fix_unicode(paragraph.text))
        document.extend(words)

    our_bigrams = defaultdict(list)
    for prev, curr in zip(document, document[1:]):
        our_bigrams[prev].append(curr)

    our_trigrams = defaultdict(list)
    starts = []
    for prev, curr, nxt in zip(document, document[1:], document[2:]):
        if prev == ".":
            starts.append(curr)
        our_trigrams[(prev, curr)].append(nxt)

    print "Bigrams :"
    for _ in range(5):
        print generate_gibberish_using_bigrams(our_bigrams)

    print
    print "Trigrams :"
    for _ in range(5):
        print generate_gibberish_using_trigrams(our_trigrams, starts)

    # topic modeling
    print
    documents = [
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

    K = 4  # topics to find
    document_topic_counts = [Counter() for _ in documents]  # number of words from each topic assigned to each document => document_topic_counts[doc_index][topic_index]
    topic_word_counts = [Counter() for _ in range(K)]       # number of assignments of each word to each topic => topic_word_counts[topic_index]["word"]
    topic_counts = [0 for _ in range(K)]                    # total number of words assigned to each topic
    document_lengths = map(len, documents)                  # total number of words in each document
    distinct_words = set(word for document in documents for word in document)
    W = len(distinct_words)
    D = len(documents)

    random.seed(0)
    document_topics = [[random.randrange(K) for word in document] for document in documents]  # random setup

    for d in range(D):
        for word, topic in zip(documents[d], document_topics[d]):
            document_topic_counts[d][topic] += 1
            topic_word_counts[topic][word] += 1
            topic_counts[topic] += 1

    # Gibbs sampling
    for _ in range(1000):
        for d in range(D):
            for i, (word, topic) in enumerate(zip(documents[d], document_topics[d])):
                # remove word and topic
                document_topic_counts[d][topic] -= 1
                topic_word_counts[topic][word] -= 1
                topic_counts[topic] -= 1
                document_lengths[d] -= 1

                # choose new topic
                new_topic = choose_new_topic(K, document_topic_counts, document_lengths, W, topic_word_counts, topic_counts, d, word)
                document_topics[d][i] = new_topic

                # add it back to the counts
                document_topic_counts[d][new_topic] += 1
                topic_word_counts[new_topic][word] += 1
                topic_counts[new_topic] += 1
                document_lengths[d] += 1

    for k, word_counts in enumerate(topic_word_counts):
        for word, count in word_counts.most_common():
            if count > 0:
                print k, word, count

    print

    topic_names = ["Big Data and programming languages",
                   "Python and statistics",
                   "databases",
                   "machine learning"]

    for document, topic_counts in zip(documents, document_topic_counts):
        print document
        for topic, count in topic_counts.most_common():
            if count > 0:
                print topic_names[topic], count,
        print
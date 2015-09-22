#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from collections import defaultdict, Counter
import glob
import pprint
import random
import re
import math
from machine_learning import split_data, precision, recall, f1_score


def tokenize(message):
    message = message.lower()
    all_words = re.findall(r"[a-z0-9']+", message)
    return set(all_words)


def count_words(training_set):
    """training_set is a list of pairs (message, is_spam)"""
    counts = defaultdict(lambda: [0, 0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts


def word_probabilities(counts, total_spams, total_non_spams, k=0.5):
    """returns list of triplets (word, p(word | spam), p(word | ~spam))"""
    return [(word, (spam + k) / (total_spams + 2 * k), (non_spam + k) / (total_non_spams + 2 * k))
            for word, (spam, non_spam) in counts.iteritems()]


def spam_probability(word_probs, message):
    words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0

    for word, prob_if_spam, prob_if_not_spam in word_probs:
        if word in words:
            # if the current word appears in the message, add the log probability of seeing it for spam / not spam
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_not_spam += math.log(prob_if_not_spam)
        else:
            # if it doesn't appear, add the log probability of *not* seeing it, ie. log(1 - prob. of seeing it)
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

    probability_if_its_spam = math.exp(log_prob_if_spam)
    probability_if_its_not_spam = math.exp(log_prob_if_not_spam)
    return probability_if_its_spam / (probability_if_its_spam + probability_if_its_not_spam)


def proba_spam_given_word(word_probability):
    """p(spam | message contains word) computed with Bayes's theorem simplification"""
    _, prob_if_spam, prob_if_not_spam = word_probability
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)


class NaiveBayesClassifier(object):
    def __init__(self, k=0.5):
        self.k = k
        self.word_probabilities = []

    def train(self, training_set):
        num_spams = [is_spam for _, is_spam in training_set].count(True)
        num_non_spams = len(training_set) - num_spams

        word_counts = count_words(training_set)
        self.word_probabilities = word_probabilities(word_counts, num_spams, num_non_spams, self.k)

    def classify(self, message):
        return spam_probability(self.word_probabilities, message)


if __name__ == '__main__':
    path = r"D:\workspace\data-science-from-scratch\spam-assassin-public-corpus\*\*"

    data = []

    for file_name in glob.glob(path):
        is_spam = 'ham' not in file_name

        with open(file_name, 'r') as mail_file:
            for line in mail_file:
                if line.startswith("Subject:"):
                    subject = re.sub(r"^Subject: ", "", line).strip()  # remove "Subject: "
                    data.append((subject, is_spam))

    # pprint.pprint(data)
    random.seed(0)
    train_data, test_data = split_data(data, 0.75)

    classifier = NaiveBayesClassifier()
    classifier.train(train_data)

    # (subject line, is it spam ?, probability that it is spam for our classifier)
    classified = [(subject, is_spam, classifier.classify(subject)) for subject, is_spam in test_data]

    # assume spam probability > 0.5 means it's predicted as spam
    counts = Counter((is_spam, spam_probability > 0.5) for _, is_spam, spam_probability in classified)
    true_positive = counts.get((True, True), 0.0)
    false_positive = counts.get((False, True), 0.0)
    false_negative = counts.get((True, False), 0.0)
    true_negative = counts.get((False, False), 0.0)

    print true_positive, false_positive, false_negative, true_negative

    assert (true_positive + false_positive + false_negative + true_negative) == len(classified)

    print precision(true_positive, false_positive, false_negative, true_negative)
    print recall(true_positive, false_positive, false_negative, true_negative)
    print f1_score(true_positive, false_positive, false_negative, true_negative)

    # sort by increasing spam probability
    classified.sort(key=lambda row: row[2])

    # 5 most problematic false positives
    spammiest_hams = filter(lambda row: not row[1], classified)[-5:]
    pprint.pprint(spammiest_hams)

    # 5 most problematic false negatives
    hammiest_spams = filter(lambda row: row[1], classified)[:5]
    pprint.pprint(hammiest_spams)

    # looks at spammiest words
    words = sorted(classifier.word_probabilities, key=proba_spam_given_word)

    print "spammiest words:", [w for w, _, __ in words[-5:]]
    print "hammiest words:", [w for w, _, __ in words[:5]]

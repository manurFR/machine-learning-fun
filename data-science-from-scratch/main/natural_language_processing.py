#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict
import random
import re
from bs4 import BeautifulSoup
import requests


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

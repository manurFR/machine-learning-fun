#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import Counter
import json
import pprint
import re
from time import sleep
from bs4 import BeautifulSoup
import requests
from twython import Twython

if __name__ == '__main__':
    def is_video(td):
        """if it has only one pricelabel whose text is Video"""
        pricelabels = td('span', 'pricelabel')
        return len(pricelabels) == 1 and pricelabels[0].text.strip().startswith("Video")

    def book_info(td):
        title = td.find('div', 'thumbheader').a.text
        by_author = td.find('div', 'AuthorName').text
        authors = [x.strip() for x in re.sub(r'^By ', '', by_author).split(',')]
        isbn_link = td.find('div', 'thumbheader').a.get('href')
        isbn = re.match(r'/product/(.*)\.do', isbn_link).groups()[0]
        date = td.find('span', 'directorydate').text.strip()

        return {'title': title, 'authors': authors, 'isbn': isbn, 'date': date}

    def get_year(book):
        return int(book['date'].split()[1])

    # oreilly = "http://shop.oreilly.com/category/browse-subjects/data.do?sortby=publicationDate&page="
    #
    # soup = BeautifulSoup(requests.get(oreilly + '1').text, 'html5lib')
    # pattern = r'Page 1 of (\d*)'
    # page_option = soup.find('option', text=re.compile(pattern)).text.strip()
    # num_pages = int(re.search(pattern, page_option).group(1))
    # print num_pages, "pages to fetch"
    #
    # books = []
    # for page_num in range(1, num_pages + 1):
    #     print "Souping page", page_num, "-", len(books), "books found so far"
    #     soup = BeautifulSoup(requests.get(oreilly + str(page_num)).text, 'html5lib')
    #
    #     for td in soup('td', 'thumbtext'):
    #         if not is_video(td):
    #             books.append(book_info(td))
    #
    #     sleep(30)
    #
    # print "Total books found:", len(books)
    # pprint.pprint(books)
    #
    # year_counts = Counter(get_year(book) for book in books if get_year(book) <= 2014)
    #
    # import matplotlib.pyplot as plt
    # years = sorted(year_counts)
    # book_counts = [year_counts[year] for year in years]
    # plt.plot(years, book_counts)
    # plt.ylabel('Number of data books')
    # plt.title('Data is big!')
    # plt.show()

    # Twitter
    with open('../../../twitter.json') as f:
        auth = json.load(f)

    twitter = Twython(auth["consumer_key"], auth["consumer_secret"])

    for status in twitter.search(q='"data science"')["statuses"]:
        user = status["user"]["screen_name"].encode('utf-8')
        text = status["text"].encode('utf-8')
        print user, ':', text
        print

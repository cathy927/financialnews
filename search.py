from __future__ import division
import re
import operator
import json
from collections import Counter
import math
import sys
import numpy
from nltk.corpus import stopwords
import string
from collections import defaultdict
from numpy import *

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


fname = 'data/stream_market.json'
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['RT', 'via']
search_word = sys.argv[1]  # pass a term as a command-line argument
count_search = Counter()
with open(fname, 'r') as f:
    count_hash = Counter()
    count_only = Counter()
    count_stop_single = Counter()
    com_max = []
    # Word co-occcurence
    com = defaultdict(lambda: defaultdict(int))

    for line in f:
        tweet = json.loads(line)
        terms_only = [term for term in preprocess(tweet['text'])
                      if term not in stop
                      and not term.startswith(('#', '@'))]
        if search_word in terms_only:
            count_search.update(terms_only)


print("Co-occurrence for %s:" % search_word)
print(count_search.most_common(20))
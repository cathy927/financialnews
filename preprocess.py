from __future__ import division
import re
import operator
from collections import Counter
import math
import sys
import numpy
from nltk.corpus import stopwords
import string
from collections import defaultdict
from numpy import *
import pandas
import json
import vincent


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


fname = 'data/stream_aapl.json'

# Count
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['via']

nfile = open('data/negative-words.txt')
nlines = nfile.readlines()
negative_vocab=[]
for line in nlines:
    line = line.strip('\n')
    negative_vocab.append(line)
nfile.close()

pfile = open('data/positive-words.txt')
plines = pfile.readlines()
positive_vocab=[]
for line in plines:
    line = line.strip('\n')
    positive_vocab.append(line)
pfile.close()

with open(fname, 'r') as f:
    count_hash = Counter()
    count_only = Counter()
    count_stop_single = Counter()
    com_max = []
    # Word co-occcurence
    com = defaultdict(lambda: defaultdict(int))
    dates_cnbc = []
    for line in f:
        tweet = json.loads(line)
        if tweet.get('text'):
            temp_str = tweet['text']
            # regex to remove unicode
            temp_str = re.sub(r'[^\x00-\x7F]+', '', temp_str)
            terms_all = [term for term in preprocess(temp_str)]
            # Count terms only once, equivalent to Document Frequency
            terms_single = set(terms_all)
            # Count hashtags only
            terms_hash = [term for term in preprocess(temp_str)
                          if term.startswith('#')]
            # Count terms only (no hashtags, no mentions)
            terms_only = [term for term in preprocess(temp_str)
                          if term not in stop and
                          not term.startswith('@')]
            #print(terms_only)
            terms_stop = [term for term in preprocess(temp_str)
                          if term not in stop]
        # track when the hashtag is mentioned
        if '#cnbc' in terms_hash:
            dates_cnbc.append(tweet['created_at'])

        # mind the ((double brackets))
        # startswith() takes a tuple (not a list) if
        # we pass a list of inputs
        count_hash.update(terms_hash)
        count_only.update(terms_only)
        count_stop_single.update(terms_stop)

        terms_only = terms_stop
        # Build co-occurrence matrix
        for i in range(len(terms_only) - 1):
            for j in range(i + 1, len(terms_only)):
                w1, w2 = sorted([terms_only[i], terms_only[j]])
                if w1 != w2:
                    com[w1][w2] += 1

    # Print the first 5 most frequent words
    print("the first 5 most frequent words: ", count_only.most_common(5))
    print("the first 5 most frequent hash:", count_hash.most_common(5))

    #visualisation
    # a list of "1" to count the hashtags
    ones = [1] * len(dates_cnbc)
    print ("ones: ", ones)
    # the index of the series
    idx = pandas.DatetimeIndex(dates_cnbc)
    print ("idx: ", idx)
    # the actual series (at series of 1s for the moment)
    cnbc = pandas.Series(ones, index=idx)

    # Resampling / bucketing
    per_minute = cnbc.resample('1S', how='sum').fillna(0)

    #print("10:", count_hash.most_common(10))
    # For each term, look for the most common co-occurrent terms
    for t1 in com:
        t1_max_terms = sorted(com[t1].items(), key=operator.itemgetter(1), reverse=True)
        for t2, t2_count in t1_max_terms:
            com_max.append(((t1, t2), t2_count))
    # Get the most frequent co-occurrences
    terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
    print("the first 10 most frequent pairs:", terms_max[:15])  #track the frequencies over time
    #To put the time series in a plot with Vincent
    print("dates_cnbc:", dates_cnbc)
    print("cnbc:", cnbc)
    print("per_minute:", per_minute)
    time_chart = vincent.Line(per_minute)
    time_chart.axis_titles(x='Time', y='Freq')
    time_chart.to_json('time_chart.json')

    #DF
    p_t = {}
    p_t_com = defaultdict(lambda: defaultdict(int))

    for term, n in count_stop_single.items():
        n_docs = 5911  # n_docs is the total n. of tweets
        p_t[term] = n / n_docs
        for t2 in com[term]:
            p_t_com[term][t2] = com[term][t2] / n_docs

    pmi = defaultdict(lambda: defaultdict(int))
    #print("p_t[increase]:", p_t['increase'])          #0.0006767044493317544   #Increase:0.0003383522246658772
    #print("p_t[decrease]:", p_t['decrease'])      #0.0001691761123329386
    #print("p_t[#NASDAQ]: ", p_t['#NASDAQ'])
    #print("com[increase]:", com['increase'])    #obama:1
    #print("com[decrease]:", com['decrease'])  # 0.0001691761123329386
    #print("com[#NASDAQ]: ", com['#NASDAQ'])
    for t1 in p_t:
        for t2 in com[t1]:
            denom = p_t[t1] * p_t[t2]
            pmi[t1][t2] = log2(p_t_com[t1][t2] / denom)

    #print("semantic_orientation: ", pmi['#ethereum']['decrease'])
    #print("semantic_orientation: ", pmi['decrease']['#ethereum'])
    #print("semantic_orientation: ", pmi)     #FMCG,#Sensex=10.94422400452973   #Sensex,#FMCG = 0
    semantic_orientation = {}
    test = 0
    test2 = 0
    for term, n in p_t.items():
        positive_assoc = sum(pmi[term][tx]+pmi[tx][term] for tx in positive_vocab)
        negative_assoc = sum(pmi[term][tx]+pmi[tx][term] for tx in negative_vocab)
        #if term == '#EUR':
        #    for tx in positive_vocab:
        #        if pmi[term][tx] <> 0 or pmi[tx][term] <> 0:
        #            test2 = test2 + pmi[term][tx]
        #            print('--positive_assoc--', tx, 'xxxxxxxx', pmi[term][tx]+pmi[tx][term])
        #positive_assoc = test2

        #if term == '#EUR':
        #    for tx in negative_vocab:
        #        if pmi[term][tx] <> 0 or pmi[tx][term] <> 0:
        #            test = test + pmi[term][tx]
        #            print('--negative_assoc--', tx, 'xxxxxxxx', pmi[term][tx]+pmi[tx][term])
        #negative_assoc = test
        semantic_orientation[term] = positive_assoc - negative_assoc

    #print('pmi[#NASDAQ][Drop]', pmi['#NASDAQ']['Drop'])
    #print('pmi[drop][#NASDAQ]', pmi['drop']['#NASDAQ'])
    semantic_sorted = sorted(semantic_orientation.items(),
                             key=operator.itemgetter(1),
                             reverse=True)
    top_pos = semantic_sorted[:10]
    top_neg = semantic_sorted[-10:]

    print("top pos: ", top_pos)
    print("top neg: ", top_neg)
    print("#cnbc: %f" % semantic_orientation['#cnbc'])
    print("unemployment: %f" % semantic_orientation['unemployment'])
    print("Trump: %f" % semantic_orientation['Trump'])

import json
import re
import operator
import datetime
import numpy
from textblob import TextBlob
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import os, sys, codecs
from nltk import bigrams
import pandas
import json
import vincent
import tushare as ts
import matplotlib.pyplot as plt
from pandas_datareader import data, wb
from pandas_highcharts.core import serialize
from pandas.compat import StringIO



reload(sys)
sys.setdefaultencoding('utf-8')
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
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

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


punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via']
fname = 'data/stream_aapl.json'
#fname = 'data/stream_GOOG.json'
with open(fname, 'r') as f:
    lis = []
    neg = 0.0
    n = 0.0
    net = 0.0
    pos = 0.0
    p = 0.0
    count_all = Counter()
    cout = 0
    tweet_time = []
    sentiment_value = []
    for line in f:
        tweet = json.loads(line)
        # Create a list with all the terms
        if tweet.get('text'):
            tweet_time.append(tweet['created_at'])
            blob = TextBlob(tweet["text"])
            cout += 1
            lis.append(blob.sentiment.polarity)
            # print blob.sentiment.subjectivity
            # print (os.listdir(tweet["text"]))
            if blob.sentiment.polarity < 0:
                sentiment = "negative"
                #print "Negative tweets: ", blob, " , ", blob.sentiment.polarity
                sentiment_value.append(blob.sentiment.polarity)
                neg += blob.sentiment.polarity
                n += 1
            elif blob.sentiment.polarity == 0:
                sentiment = "neutral"
                sentiment_value.append(0)
                net += 1
            else:
                sentiment = "positive"
                #print "Positive tweets: ", blob, " , ", blob.sentiment.polarity
                sentiment_value.append(blob.sentiment.polarity)
                pos += blob.sentiment.polarity
                p += 1

    #Chinese
    #df = ts.get_hist_data('000875', start='2017-07-29', end='2017-08-07')
    #df.to_excel('stock_sh.xlsx')
    #print ("df.date: ", df.iat[2,0])
    #df.Close.plot()

    #America
    # We will look at stock prices over the past year, starting at January 1, 2016
    start = datetime.datetime(2017, 8, 1)
    #start = datetime.datetime(2017, 8, 14)
    end = datetime.datetime(2017, 8, 28)
    # Let's get Apple stock data; Apple's ticker symbol is AAPL
    # First argument is the series we want, second is the source ("yahoo" for Yahoo! Finance), third is the start date, fourth is the end date
    apple = data.DataReader("AAPL", "yahoo", start, end)
    #apple = data.DataReader("GOOG", "yahoo", start, end)
    print ("apple: ", apple)
    df = apple

    # the index of the series
    idx1 = pandas.DatetimeIndex(tweet_time)
    # the actual series (at series of 1s for the moment)
    graph12 = pandas.Series(sentiment_value, index=idx1)
    print("graph1: ", graph12)
    # Resampling / bucketing
    print ("sentiment_value: ",sentiment_value)
    per_time1 = graph12.resample('1D', how='sum').fillna(0)
    idx1 = per_time1.index
    print ("per_time1: ", per_time1)
    graph12 = pandas.DataFrame(list(per_time1.iteritems()), columns=['Date', 'sentiment'])
    #graph12.sort_values('date', inplace=True)
    #graph12['date'] = graph12['date'].astype('category')
    #pydate_array = idx1.to_pydatetime()
    #print("pydate_array: ", pydate_array)
    #date_only_array = vectorize(lambda s: s.strftime('%Y-%m-%d'))(pydate_array)
    #idx12 = pandas.Series(date_only_array)
    #graph12['date'].cat.set_categories(idx1, inplace=True)
    print ("graph12: ", graph12.dtypes)

    stock_value = df['Close']
    idx2 = per_time1.index
    graph2 = pandas.Series(stock_value, index=idx2)
    print("idx2: ", idx2)
    print ("stock_value: ", stock_value)
    graph2 = pandas.DataFrame(list(stock_value.iteritems()), columns=['Date', 'Close'])
    #graph2.sort_values('date', inplace=True)
    #graph2['date'] = graph2['date'].astype('category')
    #pydate_array = idx2.to_pydatetime()
    #print("pydate_array: ",pydate_array)
    #date_only_array = vectorize(lambda s: s.strftime('%Y-%m-%d'))(pydate_array)
    #idx2 = pandas.Series(date_only_array)
    #graph2['date'].cat.set_categories(idx2, inplace=True)
    print ("graph12: ", graph12.dtypes)
    print ("graph2: ", graph2.dtypes)
    print ("graph12: ", graph12)
    print ("graph2: ", graph2)

    #graph12['Date'] = graph12['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    new_df = pandas.merge(graph12,graph2,how='left')
    print ("new_df: ", new_df)

    #graph2.sort_values('date', ascending=True)
    #print ("graph2: ", graph2.dtypes)
    #graph2['date'] = pandas.to_datetime(graph2['date'])
    #graph2.set_index("date", inplace=True)
    #per_time2 = graph2.resample('1D', how='sum').fillna(0)
    #print ("per_time2: ", per_time2)

    print ("per_time1: ", per_time1)
    print ("graph2: ", graph2)
    # all the data together
    #match_data = new_df
    # we need a DataFrame, to accommodate multiple series
    #all_matches = pandas.DataFrame(data=match_data, index=idx2)
    # Resampling as above
    #all_matches = all_matches.resample('1D', how='sum').fillna(0)
    all_index = new_df['Date']
    print ("all_index: ",all_index)
    all_matches = new_df.fillna(0)
    print ("all_matches:", all_matches)
    all_matches['Date'] = all_matches['Date'].apply(pandas.to_datetime)
    all_matches = all_matches.set_index('Date')
    print ("all_matches:", all_matches)
    # and now the plotting
    time_chart = vincent.Line(all_matches[['sentiment', 'Close']])
    time_chart.axis_titles(x='Time', y='Value')
    time_chart.legend(title='Matches')
    time_chart.to_json('time_chart.json')

    # output sentiment
    print "Total tweets", len(lis)
    print "no. of positive ", p
    print "Positive ", float(p / cout) * 100, "%"
    print "Negative ", float(n / cout) * 100, "%"
    print "Neutral ", float(net / len(lis)) * 100, "%"
    # print lis
    # determine if sentiment is positive, negative, or neutral

    # output sentiment
    # print sentiment

    #ax = plt.gca()
    #ax.invert_xaxis()
    #plt.show()

    #highcharts
    df = all_matches  # create your dataframe here
    chart = serialize(df, render_to='container',  output_type='json')
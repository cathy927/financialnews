from __future__ import absolute_import, print_function
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json

consumer_key = 'VQJ6ttynDZM7rHaXeKhRp4X9X'
consumer_secret = 'u4s97ExCEbSj2ARaBDX7IgehQ2VA0x2VXOpRn3xkFKyrKtRPaF'
access_token = '2434797787-xF3oGv2PSiPyZzjIFk7ey4Qa9j8tCB7BknSB2eR'
access_secret = 'vpdjfwpDXSiDU3uJBf1zVA54OQyYcTss38XcFWD8FIVD9'

class MyListener(StreamListener):
    def on_data(self, data):
        try:
            with open('python.json', 'a') as f:
                f.write(data)
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    twitter_stream = Stream(auth, MyListener())
    print("Writing Tweets")
    twitter_stream.filter(track=['bollywood'])
    # print("Writing Tweets")
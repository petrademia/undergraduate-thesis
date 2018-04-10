# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 18:50:36 2018

@author: Chastine
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 18:21:11 2017

@author: Petrus
"""

from twython import Twython, TwythonError
import jsonpickle

APP_KEY = 	'qutZDvhBlGhClbKMrx9vlbVmr'
APP_SECRET = '1u3UogRkFgmYkI3JM9reQB54N50BxrcsMFN3GX2wlkQVj2mGxz'
OAUTH_TOKEN = '877015267-PhJ68hXLnEQIJLFPPeMjLh9VhQomqnBfu5P0NwLm'
OAUTH_TOKEN_SECRET = 'vVWTkiDefigEtYMsOcLddotIYlTyxSr5FFGUj2ILrLjSX'

twitter = Twython(APP_KEY, APP_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
twitter.verify_credentials()

searchQuery = 'banjir -filter:retweets'
maxTweets = 100
tweetsPerQuery = 100
fileName = 'banjir_new.json'
sinceId = None
max_id = -1
tweetCount = 0
new_tweets = twitter.search(q=searchQuery, count=tweetsPerQuery)

print('Downloading max {0} tweets'.format(maxTweets))

f = open(fileName, 'r')
f = f.readlines()

#if f != []:
#    max_id = int(f[-1].split(',')[0])

#sinceId = 929151725948362752

tweet_with_media = []

with open(fileName, 'a') as f:
    while tweetCount < maxTweets:
        try:
            if(max_id <= 0):
                if(not sinceId):
                    print("no max_id and no sinceId")
                    new_tweets = twitter.search(q=searchQuery, count=tweetsPerQuery)
                else:
                    print("no max_id but sinceId exist")
                    new_tweets = twitter.search(q=searchQuery, count=tweetsPerQuery, since_id=sinceId)
            else:
                if (not sinceId):
                    print("max_id exist but no sinceId")
                    new_tweets = twitter.search(q=searchQuery, count=tweetsPerQuery, max_id=(max_id - 1))
                else:
                    print("max_id and sinceId exist")
                    new_tweets = twitter.search(q=searchQuery, count=tweetsPerQuery, max_id=(max_id - 1), since_id=sinceId)
            if not new_tweets:
                print('No more tweets found')
                break
            for tweet in new_tweets['statuses']:
#                print(tweet['text'])
                if 'entities' in tweet:
                    if 'media' in tweet['entities']:
                        tweet_with_media.append(tweet['entities']['media'][0]['media_url'])
#                print(tweet['entities'])
#                f.write(jsonpickle.encode(tweet['id'], unpicklable=False) + ',' + jsonpickle.encode(tweet['text'], unpicklable=False) + '\n')
            tweetCount += len(new_tweets['statuses'])
            print('Downloaded {0} tweets'.format(tweetCount))
            max_id = new_tweets['statuses'][-1]['id']
        except TwythonError as e:
            print(e.error_code)
            break
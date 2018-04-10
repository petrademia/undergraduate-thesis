# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 18:21:11 2017

@author: Petrus
"""

from twython import Twython, TwythonError
from twython import TwythonStreamer
import jsonpickle

fileName = 'banjir_stream.json'

APP_KEY = 	'qkovBPNqM0WxEytJd3Rv4Wbyw'
APP_SECRET = 'Urj4rHiq3OooXWSkFPstDDAz3KtlchWF7cD3tXwIWIYy7BGtRj'
OAUTH_TOKEN = '706363321604149248-a5XdG0sfMCD5IEyTiiSVe8JK78Wvkdn'
OAUTH_TOKEN_SECRET = 'ErsUgHEro9zgHNZECI7M0ZtbLdYAR9VEqlHHpc15nHW05'

f = open(fileName, 'w')
class MyStreamer(TwythonStreamer):
    
    def on_success(self, data):
        if 'text' in data:
            f.write(jsonpickle.encode(data['id'], unpicklable=False) + ',' + jsonpickle.encode(data['text'], unpicklable=False) + '\n')
            print(data['text'])

    def on_error(self, status_code, data):
        
        print(status_code)
        f.close()
        self.disconnect()
        
stream = MyStreamer(APP_KEY, APP_SECRET,
                    OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
stream.statuses.filter(track='macet')
f.close() # try here
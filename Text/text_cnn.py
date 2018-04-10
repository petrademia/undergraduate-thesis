# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:41:19 2017

@author: Petrus
"""

import numpy as np
import re
import itertools
from collections import Counter
import json
from unicodedata import normalize

#def clean_str(string, TREC=False):
#    """
#    Tokenization/string cleaning for all datasets except for SST.
#    Every dataset is lower cased except for TREC
#    """
#    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
#    string = re.sub(r"\'s", " \'s", string) 
#    string = re.sub(r"\'ve", " \'ve", string) 
#    string = re.sub(r"n\'t", " n\'t", string) 
#    string = re.sub(r"\'re", " \'re", string) 
#    string = re.sub(r"\'d", " \'d", string) 
#    string = re.sub(r"\'ll", " \'ll", string) 
#    string = re.sub(r",", " , ", string) 
#    string = re.sub(r"!", " ! ", string) 
#    string = re.sub(r"\(", " \( ", string) 
#    string = re.sub(r"\)", " \) ", string) 
#    string = re.sub(r"\?", " \? ", string) 
#    string = re.sub(r"\s{2,}", " ", string)    
#    return string.strip() if TREC else string.strip().lower()

category = 'macet'

def clean_str(string):
    string = string.replace('"', '')
    string = string.replace('\'', '')
    string = string.replace('\\n', ' ')
    string = re.sub(r"\\u[A-Za-z0-9]*", "", string)
    string = ' '.join(re.sub("(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",string).split())
    string = string.strip('0123456789')
    string = string.lstrip(' ')
    string = string.lstrip('rt')
    string = string.lstrip('RT')
    string = string.lstrip(' ')
    empty = ""
    for el in string:
        if(not el.isdigit()):
            empty = empty+el
    empty = empty.lstrip(' ')
    return empty.strip().lower()
    
raw_dataset = open('dataset-cutted/'+category+'_stream.json', 'r')

processed_dataset = open('dataset-cutted/'+category+'_preprocessed.json', 'w')

raw_lines = raw_dataset.readlines()

for line in raw_lines:
    string = clean_str(line)
    if(string != ''):    
        processed_dataset.write(string + '\n')
    
processed_dataset.close()
    
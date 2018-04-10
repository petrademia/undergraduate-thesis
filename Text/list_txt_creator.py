# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 02:16:02 2018

@author: Chastine
"""

import numpy as np
import re
import itertools
from collections import Counter

banjir_tweets = list(open("dataset-cutted/banjir.json", "r").readlines())
kebakaran_tweets = list(open("dataset-cutted/macet.json", "r").readlines())
macet_tweets = list(open("dataset-cutted/kebakaran.json", "r").readlines())

banjir_tweets = banjir_tweets[0:50000]
kebakaran_tweets = kebakaran_tweets[0:50000]
macet_tweets = macet_tweets[0:50000]

padding_word="<PAD/>"
# Split by words

sentences = banjir_tweets + kebakaran_tweets + macet_tweets
sentences = [s.split(" ") for s in sentences]

sequence_length = max(len(x) for x in sentences)
padded_sentences = []
for i in range(len(sentences)):
    sentence = sentences[i]
    num_padding = sequence_length - len(sentence)
    new_sentence = sentence + [padding_word] * num_padding
    padded_sentences.append(new_sentence)

word_counts = Counter(itertools.chain(*padded_sentences))
# Mapping from index to word
vocabulary_inv = [x[0] for x in word_counts.most_common()]
vocabulary_inv = list(sorted(vocabulary_inv))
# Mapping from word to index
vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

file = open('allword.txt', 'w')
for item in sentences:
  file.write("%s\n" % item)
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:09:11 2017

@author: Petrus
"""

import numpy as np
import re
import itertools
from collections import Counter

def load_data_and_labels():
    
#    positive_examples = list(open("./data/rt-polarity.pos", "r", encoding='latin-1').readlines())
#    positive_examples = [s.strip() for s in positive_examples]
#    negative_examples = list(open("./data/rt-polarity.neg", "r", encoding='latin-1').readlines())
#    negative_examples = [s.strip() for s in negative_examples]
    
    banjir_tweets = list(open("dataset/banjir.json", "r").readlines())
    kebakaran_tweets = list(open("dataset/macet.json", "r").readlines())
    macet_tweets = list(open("dataset/kebakaran.json", "r").readlines())
    
    banjir_tweets = banjir_tweets[0:50000]
    kebakaran_tweets = kebakaran_tweets[0:50000]
    macet_tweets = macet_tweets[0:50000]
    
    # Split by words
    
    x_text = banjir_tweets + kebakaran_tweets + macet_tweets
    x_text = [s.split(" ") for s in x_text] 
    
#    x_text = positive_examples + negative_examples
#    x_text = [clean_str(sent) for sent in x_text]
#    x_text = [s.split(" ") for s in x_text]
    
    # Generate labels
    
    banjir_labels = [[1, 0, 0] for _ in banjir_tweets]
    kebakaran_labels = [[0, 1, 0] for _ in kebakaran_tweets]
    macet_labels = [[0, 0, 1] for _ in macet_tweets]
    
    y = np.concatenate([banjir_labels, kebakaran_labels, macet_labels], 0)

#    positive_labels = [[0, 1] for _ in positive_examples]
#    negative_labels = [[1, 0] for _ in negative_examples]
#    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def load_data():
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

sentences, labels = load_data_and_labels()
sentences_padded = pad_sentences(sentences)
vocabulary, vocabulary_inv = build_vocab(sentences_padded)
x, y = build_input_data(sentences_padded, labels, vocabulary)
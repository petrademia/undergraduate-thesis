# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 15:44:55 2018

@author: Chastine
"""


import numpy as np
import re
import itertools
from collections import Counter
import os
import random
from sklearn import preprocessing
from keras.preprocessing.text import text_to_word_sequence
from sklearn.cross_validation import train_test_split
from keras.layers import Input, Dense, Embedding, Concatenate, Convolution2D, MaxPooling2D, Dropout, Conv1D, Conv2D, LSTM, concatenate
from keras.layers.core import Flatten
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

dataset = '../dataset/text/original/'

all_tweets = {}
length_each_category = 10000
sentences = []
y_train = []
categories = []

label_binarizer = preprocessing.LabelBinarizer()

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
    processed_string = ""
    for character in string:
        if(not character.isdigit()):
            processed_string = processed_string + character
    processed_string = processed_string.lstrip(' ')
    return processed_string.strip().lower()


def pad_sentences(sentences, padding_word="<PAD/>"):
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_vocab(sentences):
    
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

for filename in os.listdir(dataset):
    category = os.path.splitext(filename)[0]
    categories.append(category)

num_classes = len(categories)

label_binarizer.fit(categories)
for filename in os.listdir(dataset):
    category = os.path.splitext(filename)[0]
    print("Processing category: " + category)
    list_of_each_file = open(dataset + filename).readlines()
    list_of_each_file = [clean_str(sentence) for sentence in list_of_each_file]
    list_of_each_file = [sentence for sentence in list_of_each_file if category in sentence]
    rand_options = list_of_each_file
    for _ in range(length_each_category):
        rand_sentence = random.choice(rand_options)
        sentences.append(rand_sentence)
        rand_options.remove(rand_sentence)
        y_train.append(category)
    all_tweets[category] = list_of_each_file
    
y = label_binarizer.transform(y_train)

sentences_tokenized = [sentence.split(' ') for sentence in sentences]

sentence_padded = pad_sentences(sentences_tokenized)
vocabulary, vocabulary_inv = build_vocab(sentence_padded)

x = np.array([[vocabulary[word] for word in sentence] for sentence in sentence_padded])

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

vocabulary_size =  len(vocabulary)
sequence_length = x.shape[1]

filepath = "weights/lstm/embedding_cnn_lstm/weights@epoch-{epoch:03d}-{val_acc:.2f}.hdf5"

tensorboard = TensorBoard(log_dir='./Graph/embedding_cnn_lstm', histogram_freq=0,  
          write_graph=True, write_images=True)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=5)

callbacks = [tensorboard, checkpoint]

#callbacks = [tensorboard, checkpoint, early_stopping]

convs = []
filter_sizes = [3, 4, 5]

for filter_size in filter_sizes:
    conv = Sequential()
    conv.add(Conv1D(64, filter_size, strides = 1, activation = 'relu', input_shape = (30, 128)))
    convs.append(conv)

model = Sequential()
model.add(Embedding(vocabulary_size, 128, input_length=sequence_length))
model.add((LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test), callbacks = callbacks)

main_input = Input(shape=(sequence_length,), dtype='int32')
embedding_layer = Embedding(vocabulary_size, 128, input_length=sequence_length) (main_input)
conv_layer_fz_2 = Conv1D(128, 2, strides=1, activation='relu') (embedding_layer)
conv_layer_fz_3 = Conv1D(128, 3, strides=1, activation='relu') (embedding_layer)
conv_layer_fz_4 = Conv1D(128, 4, strides=1, activation='relu') (embedding_layer)
concatenate_layer = concatenate([conv_layer_fz_2, conv_layer_fz_3, conv_layer_fz_4], axis = 1) 
lstm_layer = LSTM(128, dropout=0.2, recurrent_dropout=0.2) (concatenate_layer)
output = Dense(num_classes, activation='softmax') (lstm_layer)
model = Model(inputs=main_input, outputs=output)
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test), callbacks = callbacks)
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:20:33 2018

@author: Chastine
"""

from keras.layers import Input, Dense, Embedding, merge, Convolution2D, MaxPooling2D, Dropout
from sklearn.cross_validation import train_test_split
from keras.layers.core import Reshape, Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from data_helpers import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Bidirectional
from sklearn.manifold import TSNE
from keras.preprocessing.text import text_to_word_sequence
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from gensim.models import FastText


model = FastText.load_fasttext_format('model')

print ('Loading data')
x, y, vocabulary, vocabulary_inv = load_data()

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)
text_classifier = load_model('lstm/weights@epoch-003-1.00.hdf5')

embedding_weights = text_classifier.get_weights()[0]

embedding = np.empty((len(vocabulary_inv), 128), dtype=np.float32)
for i, word in enumerate(vocabulary_inv):
    embedding[i] = embedding_weights[i]
    
layer_name = 'embedding_2'
intermediate_layer_model = Model(inputs=text_classifier.input,
                                 outputs=text_classifier.get_layer(layer_name).output)
intermediate_X_train = intermediate_layer_model.predict(X_train)
intermediate_X_test = intermediate_layer_model.predict(X_test)



filepath="lstm-without-embedding/weights@epoch-{epoch:03d}-{val_acc:.2f}.hdf5"

tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)

early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=5)

callbacks = [tensorboard, checkpoint, early_stopping]


model = Sequential()
model.add((LSTM(128, dropout=0.2, recurrent_dropout=0.2, batch_input_shape=(None, 30, 128))))
model.add(Dense(3, activation='softmax'))
model.summary()



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(intermediate_X_train, y_train, batch_size=32, epochs=1000, validation_data = (intermediate_X_test, y_test), callbacks = callbacks)

try_to_predict = np.empty((1, 30, 128), dtype=np.float32)

text_classifier = load_model('lstm-without-embedding/weights@epoch-008-1.00.hdf5')

def predict_text(sentence):
    
    padding_word="<PAD/>"
    sequence_length = 30
    to_predict = np.empty((30, 128), dtype=np.float32)
    
    sentence = sentence.split(" ")
    num_padding = sequence_length - len(sentence)
    new_sentence = sentence + [padding_word] * num_padding
    for index, word in enumerate(new_sentence):
#        print(index)
        if word in vocabulary:
            to_predict[index] = embedding[vocabulary[word]]
        else:
            to_predict[index] = model.wv[word]
            
    to_predict = np.expand_dims(to_predict, axis = 0)
    return to_predict

x = predict_text('kebakaran rumah terjadi karena tabung gas')
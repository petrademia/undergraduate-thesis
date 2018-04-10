# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 03:15:07 2018

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

print ('Loading data')
x, y, vocabulary, vocabulary_inv = load_data()

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

text_classifier = load_model('lstm/weights@epoch-003-1.00.hdf5')
embedding_weights = text_classifier.get_weights()[0]

embedding = np.empty((len(vocabulary_inv), 128), dtype=np.float32)
for i, word in enumerate(vocabulary_inv):
    embedding[i] = embedding_weights[i]

# setup a TensorFlow session
tf.reset_default_graph()
sess = tf.InteractiveSession()
X = tf.Variable([0.0], name='embedding')
place = tf.placeholder(tf.float32, shape=embedding.shape)
set_x = tf.assign(X, place, validate_shape=False)
sess.run(tf.global_variables_initializer())
sess.run(set_x, feed_dict={place: embedding})

# write labels
with open('log/metadata.tsv', 'w') as f:
    for word in vocabulary_inv:
        f.write(word + '\n')

# create a TensorFlow summary writer
summary_writer = tf.summary.FileWriter('log', sess.graph)
config = projector.ProjectorConfig()
embedding_conf = config.embeddings.add()
embedding_conf.tensor_name = 'embedding:0'
embedding_conf.metadata_path = os.path.join('log', 'metadata.tsv')
projector.visualize_embeddings(summary_writer, config)

# save the model
saver = tf.train.Saver()
saver.save(sess, os.path.join('log', "model.ckpt"))
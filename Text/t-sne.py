# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 06:58:51 2018

@author: Chastine
"""

import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from gensim.models import FastText

from gensim.models import FastText
model = FastText.load_fasttext_format('model')



# create a list of vectors
embedding = np.empty((len(model.wv.vocab), model.vector_size), dtype=np.float32)
for i, word in enumerate(model.wv.vocab):
    embedding[i] = model[word]

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
    for word in model.wv.vocab:
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